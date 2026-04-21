import modal

app = modal.App("example-flux2")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
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
        "nvtop"
    )
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "git+https://github.com/huggingface/diffusers",
        "transformers",
        "accelerate",
        "pillow",
        "sentencepiece",
        "protobuf",
    )
)

N_GPU = 1
MINUTES = 60
HOURS = 60 * MINUTES

MODEL_NAME = "black-forest-labs/FLUX.2-dev"
LORA_REPO = "fal/FLUX.2-dev-Turbo"
LORA_WEIGHT = "flux.2-turbo-lora.safetensors"

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

CACHE_DIR = "/cache"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]

image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": str(CACHE_DIR)})


with image.imports():
    import time
    from io import BytesIO
    import torch
    from diffusers import Flux2Pipeline
    from PIL import Image


@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes=volumes,
    secrets=secrets,
    timeout=24 * HOURS,
)
class Model:
    @modal.enter()
    def enter(self):
        print(f"Loading {MODEL_NAME}...")
        start_time = time.time()
        self.device = "cuda:0"

        self.pipe = Flux2Pipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        ).to(self.device)

        print(f"Loading LoRA weights from {LORA_REPO}...")
        self.pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT)

        total_loadtime = time.time() - start_time
        print(f"Total load time: {total_loadtime:.3f}s")

    @modal.method()
    def inference(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 8,
        guidance_scale: float = 2.5,
        sigmas: list[float] | None = None,
        seed: int | None = 42,
    ) -> bytes:
        start_time = time.time()

        if sigmas is None:
            sigmas = TURBO_SIGMAS

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        pipe_start = time.time()
        result = self.pipe(
            prompt=prompt,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        pipe_latency = time.time() - pipe_start

        image = result.images[0]
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        total_latency = time.time() - start_time
        print(f"Inference latency: {total_latency:.3f}s (model: {pipe_latency:.3f}s)")

        return image_bytes


@app.local_entrypoint()
def main(
    output_path: str = None,
    prompt: str = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 2.5,
    seed: int = 42,
    num_iterations: int = 1,
):
    """
    Local entrypoint for FLUX.2 image generation on Modal.

    Args:
        output_path: Output image path (auto-generated if not provided)
        prompt: Text prompt for image generation
        width: Image width (default: 1024)
        height: Image height (default: 1024)
        num_inference_steps: Number of diffusion steps (default: 8)
        guidance_scale: Guidance scale (default: 2.5)
        seed: Random seed for reproducibility (default: 42)
        num_iterations: Number of inference runs (default: 1)
    """
    import time
    from pathlib import Path
    from uuid import uuid4

    if prompt is None:
        prompt = (
            "Industrial product shot of a chrome turbocharger with glowing hot exhaust manifold, "
            "engraved text 'FLUX.2 [dev] Turbo by fal' on the compressor housing and 'fal' on the "
            "turbine wheel, gradient heat glow from orange to electric blue, studio lighting with "
            "dramatic shadows, shallow depth of field, engineering blueprint pattern in background."
        )
    if output_path is None:
        output_path = Path(__file__).parent / f"output/{uuid4()}.png"

    if isinstance(output_path, str):
        output_path = Path(output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"Output path: {output_path}")
    print(f"Size: {width}x{height}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    for i in range(num_iterations):
        iteration_start = time.time()

        if num_iterations > 1:
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        inference_start = time.time()
        print(f"Generating image with FLUX.2 Turbo...")
        output_image_bytes = Model().inference.remote(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.3f}s")

        save_start = time.time()
        iter_output = output_path if num_iterations == 1 else output_path.with_stem(f"{output_path.stem}_{i+1}")
        print(f"Saving output image to {iter_output}")
        iter_output.write_bytes(output_image_bytes)
        save_time = time.time() - save_start
        print(f"Image saved in {save_time:.3f}s ({len(output_image_bytes) / 1024:.1f} KB)")

        iteration_time = time.time() - iteration_start

        print(f"\nLatency Breakdown (Iteration {i+1}):")
        print(f"  - Remote inference:  {inference_time:8.3f}s ({inference_time/iteration_time*100:5.1f}%)")
        print(f"  - Image saving:      {save_time:8.3f}s ({save_time/iteration_time*100:5.1f}%)")
        print(f"  - Total:             {iteration_time:8.3f}s")

    print(f"\nAll {num_iterations} iteration(s) completed successfully!")
    print(f"Final output saved to: {output_path}")
