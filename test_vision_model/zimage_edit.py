import modal

app = modal.App("example-zimage-edit")

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

MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"

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
    import time
    from io import BytesIO
    import torch
    from diffusers import ZImageImg2ImgPipeline
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

        self.pipe = ZImageImg2ImgPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            cache_dir=CACHE_DIR,
        ).to(self.device)

        total_loadtime = time.time() - start_time
        print(f"Total load time: {total_loadtime:.3f}s")

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str = None,
        width: int = 1024,
        height: int = 1664,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        strength: float = 0.6,
        seed: int | None = 42,
    ) -> bytes:
        start_time = time.time()

        init_image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((width, height))
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        pipe_start = time.time()
        result = self.pipe(
            prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        pipe_latency = time.time() - pipe_start

        output_image = result.images[0]
        buf = BytesIO()
        output_image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        total_latency = time.time() - start_time
        print(f"Inference latency: {total_latency:.3f}s (model: {pipe_latency:.3f}s)")

        return image_bytes


@app.local_entrypoint()
def main(
    image_path: str = None,
    output_path: str = None,
    prompt: str = None,
    negative_prompt: str = None,
    width: int = 1024,
    height: int = 1664,
    num_inference_steps: int = 9,
    guidance_scale: float = 2.5,
    strength: float = 0.6,
    seed: int = 42,
    num_iterations: int = 1,
):
    """
    Local entrypoint for Z-Image-Turbo image editing on Modal.

    Args:
        image_path: Input image path (required)
        output_path: Output image path (auto-generated if not provided)
        prompt: Text prompt describing the desired edit
        width: Image width (default: 1024)
        height: Image height (default: 1024)
        num_inference_steps: Number of steps — 9 results in 8 DiT forwards (default: 9)
        guidance_scale: Guidance scale — should be 0.0 for Turbo models (default: 0.0)
        strength: Edit strength — 0.0 keeps original, 1.0 fully redraws (default: 0.6)
        seed: Random seed for reproducibility (default: 42)
        num_iterations: Number of inference runs (default: 1)
    """
    import time
    from pathlib import Path
    from uuid import uuid4

    if image_path is None:
        image_path = Path(__file__).parent / "sample1.png"
    if prompt is None:
        prompt = """A realistic photo of the SAME man from the input image, preserving 100% of his original face, identity, facial features, hairstyle, expression, and pose.

He is now wearing a long-sleeve shirt, natural fabric, well-fitted, realistic wrinkles and texture.

IMPORTANT: keep the exact same face, same eyes, same nose, same mouth, same proportions, no changes to identity.

The background remains completely unchanged, identical to the original image, same lighting, same environment.

Ultra photorealistic, high detail, natural skin texture, consistent lighting, no distortion, no artifacts."""
    if output_path is None:
        output_path = Path(__file__).parent / f"output/{uuid4()}.png"

    if isinstance(image_path, str):
        image_path = Path(image_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"Input image: {image_path}")
    print(f"Output path: {output_path}")
    print(f"Size: {width}x{height}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Strength: {strength}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    for i in range(num_iterations):
        iteration_start = time.time()

        if num_iterations > 1:
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        load_start = time.time()
        print(f"Loading input image from {image_path}")
        input_image_bytes = image_path.read_bytes()
        load_time = time.time() - load_start
        print(f"Image loaded in {load_time:.3f}s ({len(input_image_bytes) / 1024:.1f} KB)")

        inference_start = time.time()
        print(f"Editing image with Z-Image-Turbo...")
        output_image_bytes = Model().inference.remote(
            image_bytes=input_image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
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
