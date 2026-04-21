import modal

app = modal.App("example-qwen-image-edit")

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

MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"

CACHE_DIR = "/cache"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]

image = image.env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_CACHE": str(CACHE_DIR),
    "HF_HUB_DOWNLOAD_TIMEOUT": "6000",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})


with image.imports():
    import os
    import time
    from io import BytesIO
    import torch
    from diffusers import QwenImageEditPlusPipeline
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

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(disable=None)

        total_loadtime = time.time() - start_time
        print(f"Total load time: {total_loadtime:.3f}s")

    @modal.method()
    def inference(
        self,
        image_bytes_list: list[bytes],
        prompt: str,
        negative_prompt: str = " ",
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        seed: int = 0,
    ) -> bytes:
        start_time = time.time()

        input_images = [
            Image.open(BytesIO(b)).convert("RGB")
            for b in image_bytes_list
        ]

        pipe_start = time.time()
        with torch.inference_mode():
            output = self.pipeline(
                image=input_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=torch.manual_seed(seed),
            )
        pipe_latency = time.time() - pipe_start

        output_image = output.images[0]
        buf = BytesIO()
        output_image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        total_latency = time.time() - start_time
        print(f"Inference latency: {total_latency:.3f}s (model: {pipe_latency:.3f}s)")

        return image_bytes


@app.local_entrypoint()
def main(
    image_paths: str = None,
    output_path: str = None,
    prompt: str = None,
    negative_prompt: str = " ",
    num_inference_steps: int = 40,
    guidance_scale: float = 1.0,
    true_cfg_scale: float = 4.0,
    num_images_per_prompt: int = 1,
    seed: int = 0,
    num_iterations: int = 1,
):
    """
    Local entrypoint for Qwen image editing on Modal.

    Args:
        image_paths: Comma-separated input image paths e.g. "img1.png,img2.png"
        output_path: Output image path (auto-generated if not provided)
        prompt: Text prompt describing the desired edit
        negative_prompt: Negative prompt (default: " ")
        num_inference_steps: Number of diffusion steps (default: 40)
        guidance_scale: Guidance scale (default: 1.0)
        true_cfg_scale: True CFG scale (default: 4.0)
        num_images_per_prompt: Number of output images (default: 1)
        seed: Random seed for reproducibility (default: 0)
        num_iterations: Number of inference runs (default: 1)
    """
    import time
    from pathlib import Path
    from uuid import uuid4

    if image_paths is None:
        image_paths = str(Path(__file__).parent / "input1.png") + "," + str(Path(__file__).parent / "input2.png")
    if prompt is None:
        prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
    if output_path is None:
        output_path = Path(__file__).parent / f"output/{uuid4()}.png"

    if isinstance(output_path, str):
        output_path = Path(output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)

    input_paths = [Path(p.strip()) for p in image_paths.split(",")]

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"Input images: {[str(p) for p in input_paths]}")
    print(f"Output path: {output_path}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    for i in range(num_iterations):
        iteration_start = time.time()

        if num_iterations > 1:
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        load_start = time.time()
        print(f"Loading input images...")
        input_image_bytes_list = [p.read_bytes() for p in input_paths]
        load_time = time.time() - load_start
        print(f"Images loaded in {load_time:.3f}s ({sum(len(b) for b in input_image_bytes_list) / 1024:.1f} KB total)")

        inference_start = time.time()
        print(f"Editing images with Qwen Image Edit...")
        output_image_bytes = Model().inference.remote(
            image_bytes_list=input_image_bytes_list,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            num_images_per_prompt=num_images_per_prompt,
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
        print(f"  - Image loading:     {load_time:8.3f}s ({load_time/iteration_time*100:5.1f}%)")
        print(f"  - Remote inference:  {inference_time:8.3f}s ({inference_time/iteration_time*100:5.1f}%)")
        print(f"  - Image saving:      {save_time:8.3f}s ({save_time/iteration_time*100:5.1f}%)")
        print(f"  - Total:             {iteration_time:8.3f}s")

    print(f"\nAll {num_iterations} iteration(s) completed successfully!")
    print(f"Final output saved to: {output_path}")
