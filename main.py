import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .run_commands("apt-get update")
    .run_commands(
        "apt-get install -y bash "
        "build-essential "
        "git "
        "git-lfs "
        "curl "
        "ca-certificates "
        "libglib2.0-0 "
        "libsndfile1-dev "
        "libgl1 "
        "nvtop "
        "libnuma1"
    )
    .pip_install(
        "psutil",
        "GPUtil",
        "requests",
        "litellm[proxy]",
        "PyYAML",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands(
        'uv pip install --system '
        '"sglang" --prerelease=allow'
    )
)
app = modal.App(name="sglang-qwen35", image=image)



with image.imports():
    import time
    import subprocess
    import requests

volume = modal.Volume.from_name("qwen", create_if_missing=True)


image = image.env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_DOWNLOAD_TIMEOUT": "6000",
    "HF_HUB_ETAG_TIMEOUT": "120",
    # torch.compile cache – survives container restarts when volume is mounted
})


MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
SGLANG_PORT = 8000
LITELLM_PORT = 4000
N_GPU = 1
MINUTES = 60
HOURS = 60 * MINUTES
MODELS_DIR = "/qwen"
FAST_BOOT = False


def _start_sglang():
    """Start SGLang server as a subprocess and wait until ready."""
    import os
    model_path = MODELS_DIR + f"/{MODEL_NAME}"

    # Verify model files exist
    if not os.path.isdir(model_path):
        entries = os.listdir(MODELS_DIR) if os.path.isdir(MODELS_DIR) else "MODELS_DIR not found"
        raise FileNotFoundError(
            f"Model directory not found: {model_path}. "
            f"Contents of {MODELS_DIR}: {entries}. "
            f"Run 'modal run main.py::download_model' first."
        )
    print(f"Model directory found: {model_path}, contents: {os.listdir(model_path)[:10]}")

    sglang_binary = "python"

    cmd = [
        sglang_binary,
        "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(SGLANG_PORT),
    ]

    if FAST_BOOT:
        cmd += ["--disable-cuda-graph"]

    cmd += ["--reasoning-parser", "qwen3"]
    cmd += ["--tool-call-parser", "qwen3_coder"]

    if N_GPU > 1:
        cmd += [f"--tp {N_GPU}"]

    print(f"Starting SGLang server with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)

    print("Waiting for SGLang server to be ready...")
    max_retries = 240
    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{SGLANG_PORT}/health", timeout=5)
            if response.status_code == 200:
                print("SGLang server is ready!")
                return process
            if i % 12 == 0:
                print(f"Waiting for SGLang server (status={response.status_code})... ({i * 5}s elapsed)")
        except Exception:
            if i % 12 == 0:
                print(f"Waiting for SGLang server... ({i * 5}s elapsed)")
        time.sleep(5)

    raise Exception("SGLang server failed to start within timeout period")


def _write_litellm_config():
    """Write litellm proxy config that routes to local sglang."""
    import yaml

    config = {
        "model_list": [
            {
                "model_name": MODEL_NAME,
                "litellm_params": {
                    "model": f"openai/{MODEL_NAME}",
                    "api_base": f"http://localhost:{SGLANG_PORT}/v1",
                    "api_key": "no-key-needed",
                },
            }
        ],
    }
    config_path = "/tmp/litellm_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def _start_litellm(config_path):
    """Start litellm proxy as a subprocess and wait until ready."""
    cmd = [
        "litellm",
        "--config", config_path,
        "--host", "0.0.0.0",
        "--port", str(LITELLM_PORT),
    ]
    print(f"Starting LiteLLM proxy with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)

    print("Waiting for LiteLLM proxy to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{LITELLM_PORT}/health", timeout=5)
            if response.status_code == 200:
                print("LiteLLM proxy is ready!")
                return process
        except Exception:
            if i % 6 == 0:
                print(f"Waiting for LiteLLM proxy... ({i * 2}s elapsed)")
            time.sleep(2)

    raise Exception("LiteLLM proxy failed to start within timeout period")


@app.function(
    image=image,
    volumes={MODELS_DIR: volume},
    timeout=30 * MINUTES,
)
def download_model(
    model_name=MODEL_NAME,
    force_download=False,
):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + f"/{model_name}",
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],
        force_download=force_download,
    )

    volume.commit()
    print(f"Model {model_name} downloaded to {MODELS_DIR}/{model_name}")

@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",
    max_containers=1,
    scaledown_window=30 * MINUTES,
    timeout=24 * HOURS,
    volumes={
        MODELS_DIR: volume,
    },
)
@modal.web_server(port=LITELLM_PORT, startup_timeout=600)
def serve():
    """Start sglang backend, then litellm proxy in front of it."""
    _start_sglang()
    config_path = _write_litellm_config()
    _start_litellm(config_path)


@app.local_entrypoint()
def main():
    from openai import OpenAI
    import time

    inference_start = time.time()

    client = OpenAI(
        base_url="https://dorathekid110--sglang-qwen35-serve-dev.modal.run",
        api_key="no-key-needed",
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello!"}],
    )

    print(response)

    inference_time = time.time() - inference_start
    print(f"✓ Inference completed in {inference_time:.3f}s")
