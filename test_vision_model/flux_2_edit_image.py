import modal

app = modal.App("example-flux2-edit-image")

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
        "peft",
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
    from diffusers import Flux2Pipeline, AutoPipelineForImage2Image
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

        base_pipe = Flux2Pipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )

        # Convert to img2img reusing already-loaded components (avoids component mismatch from from_pretrained)
        self.pipe = AutoPipelineForImage2Image.from_pipe(base_pipe)

        # Stream model components to GPU only when needed, free after each step
        self.pipe.enable_model_cpu_offload(gpu_id=0)

        # Reduce transformer attention peak memory
        self.pipe.enable_attention_slicing()

        print(f"Loading LoRA weights from {LORA_REPO}...")
        self.pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT)

        total_loadtime = time.time() - start_time
        print(f"Total load time: {total_loadtime:.3f}s")

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        width: int = 1024,
        height: int = 1664,
        num_inference_steps: int = 8,
        guidance_scale: float = 2.5,
        strength: float = 0.75,
        sigmas: list[float] | None = None,
        seed: int | None = 42,
    ) -> bytes:
        start_time = time.time()

        if sigmas is None:
            sigmas = TURBO_SIGMAS

        init_image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((width, height))
        generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None

        pipe_start = time.time()
        result = self.pipe(
            prompt=prompt,
            image=init_image,
            # strength=strength,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
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
    width: int = 1024,
    height: int = 1664,
    num_inference_steps: int = 8,
    guidance_scale: float = 2.5,
    strength: float = 0.6,
    seed: int = 42,
    num_iterations: int = 1,
):
    """
    Local entrypoint for FLUX.2 image editing on Modal.

    Args:
        image_path: Input image path (required)
        output_path: Output image path (auto-generated if not provided)
        prompt: Text prompt describing the desired edit
        width: Image width (default: 1024)
        height: Image height (default: 1024)
        num_inference_steps: Number of diffusion steps (default: 8)
        guidance_scale: Guidance scale (default: 2.5)
        strength: Edit strength — 0.0 keeps original, 1.0 fully redraws (default: 0.75)
        seed: Random seed for reproducibility (default: 42)
        num_iterations: Number of inference runs (default: 1)
    """
    import time
    from pathlib import Path
    from uuid import uuid4

    if image_path is None:
        image_path = Path(__file__).parent / "sample1.png"
    if prompt is None:
        prompt = """Edit the provided image of a man sitting casually. Replace his current t-shirt with a long-sleeve button-up shirt (formal or smart casual style, well-fitted, natural fabric folds). Add black eyeglasses with a simple, modern frame on his face.
        Preserve the person’s identity, facial features, skin texture, hairstyle, pose, body proportions, and hand position exactly as in the original image. Maintain the same background, lighting, color tones, camera angle, and depth of field.
        The edit should be photorealistic and seamless, with natural shadows and reflections on the glasses. No distortion, no extra limbs, no changes to expression or posture. Keep everything identical except the clothing and the added glasses.
        """
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
        print(f"Editing image with FLUX.2 Turbo...")
        output_image_bytes = Model().inference.remote(
            image_bytes=input_image_bytes,
            prompt=prompt,
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
