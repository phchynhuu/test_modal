import modal

app = modal.App("example-ltx2")


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
        "av",
        "git+https://github.com/huggingface/diffusers",
        "transformers",
        "accelerate",
        "opencv-python",
        "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.10-cp311-cp311-linux_x86_64.whl"
    )
)

N_GPU = 1
MINUTES = 60
HOURS = 60 * MINUTES

# Download the model
MODEL_NAME = "dg845/LTX-2.3-Diffusers"
DISTIL_MODEL_NAME = "dg845/LTX-2.3-Distilled-Diffusers"
UP_SAMEPLER_MODEL_NAME = "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"

CACHE_DIR = "/cache"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]

image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": str(CACHE_DIR)})


with image.imports():
    import os
    import tempfile
    import time
    import torch
    from io import BytesIO
    from uuid import uuid4
    from pathlib import Path
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
    from PIL import Image
    from diffusers.utils import load_image
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video



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

        self.pipe = LTX2ImageToVideoPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        self.pipe.enable_sequential_cpu_offload(device=self.device)
        self.pipe.vae.enable_tiling()

        print(f"Loading upsample {UP_SAMEPLER_MODEL_NAME}...")

        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            UP_SAMEPLER_MODEL_NAME,
            subfolder="latent_upsampler",
            torch_dtype=torch.bfloat16,
        )
        self.upsample_pipe = LTX2LatentUpsamplePipeline(vae=self.pipe.vae, latent_upsampler=latent_upsampler)
        self.upsample_pipe.enable_model_cpu_offload(device=self.device)

        # Stage 2
        self.pipe.load_lora_weights(
            "Lightricks/LTX-2.3",
            adapter_name="stage_2_distilled",
            weight_name="ltx-2.3-22b-distilled-lora-384.safetensors",
        )
        self.pipe.set_adapters("stage_2_distilled", 1.0)
        # Change scheduler to use Stage 2 distilled sigmas as is
        new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
        )
        self.pipe.scheduler = new_scheduler

        total_loadtime = time.time() - start_time

        print(f"Total time: {total_loadtime:.3f}s")


    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        width: int,
        height: int,
        duration: int,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 30,
        negative_prompt: str = " ",
        seed: int | None = 42,
        frame_rate: int = 24,
    ) -> bytes:
        start_time = time.time()

        init_image = load_image(Image.open(BytesIO(image_bytes))).resize((width, height))

        pipe_start = time.time()
        generator = torch.Generator(self.device).manual_seed(seed)
        video_latent, audio_latent = self.pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=(duration * frame_rate) + 1,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
            return_dict=False,
            #
            audio_guidance_scale=5.5,
            audio_stg_scale=1.0,
            audio_modality_scale=3.0,
            audio_guidance_rescale=0.7,
            spatio_temporal_guidance_blocks=[28],
            use_cross_timestep=True,
            stg_scale=1.2,
        )

        upscaled_video_latent = self.upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        video, audio = self.pipe(
            image=image,
            latents=upscaled_video_latent,
            audio_latents=audio_latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width * 2,
            height=height * 2,
            num_frames=(duration * frame_rate) + 1,
            frame_rate=frame_rate,
            num_inference_steps=3,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,  # For Stage 2 distilled, disable all guidance
            stg_scale=0.0,
            modality_scale=1.0,
            guidance_rescale=0.0,
            audio_guidance_scale=1.0,
            audio_stg_scale=0.0,
            audio_modality_scale=1.0,
            audio_guidance_rescale=0.0,
            spatio_temporal_guidance_blocks=None,
            use_cross_timestep=True,
            generator=generator,
            output_type="np",
            return_dict=False,
        )

        # Decode upscaled latents to video frames
        # Export video to bytes via a temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

            encode_video(
                video[0],
                fps=frame_rate,
                audio=audio[0].float().cpu(),
                audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
                output_path=tmp_path,
            )
        pipe_latency = time.time() - pipe_start

        video_bytes = open(tmp_path, "rb").read()
        os.remove(tmp_path)

        total_latency = time.time() - start_time

        print(f"Inference latency: {total_latency:.3f}s (model: {pipe_latency:.3f}s)")

        return video_bytes


@app.local_entrypoint()
def main(
    image_path: str = None,  # Required for all modes
    output_path: str = None,
    prompt: str = None,
    negative_prompt: str = None,
    width: int = 512,
    height: int = 768,
    num_inference_steps: int = 40,
    guidance_scale: int = 4.0,
    seed: int = 42,
    num_iterations = 1,
    frame_rate: int = 24,
    duration: int = 5,
):
    """
    Unified entrypoint for LTX2 inference.

    Modes:

    Args:
        mode: Inference mode ("edit" or "nunchaku")
        image_path: Input image path (required for all modes)
        output_path: Output image path (auto-generated if not provided)
        prompt: Text prompt for editing
        num_iterations: Number of inference runs (default: 1)
        num_inference_steps: Number of steps (default: 12 for edit, 4 for nunchaku)
        seed: Random seed for reproducibility (default: 0)
    """

    # Set default values based on mode
    if image_path is None:
        image_path = Path(__file__).parent / "src.jpeg"
    if prompt is None:
        prompt = """The person in the input image remains the same identity, preserving facial features, hairstyle, and appearance.
            The subject starts with a neutral expression, then slowly transitions into a natural, gentle smile. After smiling, they softly wink one eye in a playful and realistic manner.
            Facial movements are subtle and lifelike, with smooth muscle motion. No exaggerated expressions, no distortion, no change in face structure.
            The head stays mostly still, with only minimal natural movement. Eyes are focused forward.
            Lighting, background, and composition remain consistent with the original image.
            Cinematic, photorealistic, high detail, smooth motion, 4K quality.
        """
    if negative_prompt is None:
        negative_prompt = "distorted face, blurry, low quality, extra limbs, deformed eyes, unnatural smile, exaggerated expression, face morphing, identity change, flickering, jitter, artifacts"
    if output_path is None:
        output_path = Path(__file__).parent / f"output/{str(uuid4)}.mp4"
    if num_inference_steps is None:
        num_inference_steps = 4

    # Convert paths to Path objects
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if image_path and isinstance(image_path, str):
        image_path = Path(image_path)

    # Create output directory
    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"Input image: {image_path}")
    print(f"Output path: {output_path}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Run inference iterations
    for i in range(num_iterations):
        iteration_start = time.time()

        if num_iterations > 1:
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        # Load input image
        load_start = time.time()
        print(f"📖 Loading input image from {image_path}")
        input_image_bytes = image_path.read_bytes()
        load_time = time.time() - load_start
        print(f"✓ Image loaded in {load_time:.3f}s ({len(input_image_bytes) / 1024:.1f} KB)")

        # Run inference
        inference_start = time.time()

        print(f"✏️  Generate video with standard model...")
        output_video_bytes = Model().inference.remote(
            image_bytes=input_image_bytes,
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            frame_rate=frame_rate,
            duration=duration,
            seed=seed,
        )
        inference_time = time.time() - inference_start
        print(f"✓ Inference completed in {inference_time:.3f}s")

        # Save output image
        save_start = time.time()
        print(f"💾 Saving output image to {output_path}")
        output_path.write_bytes(output_video_bytes)
        save_time = time.time() - save_start
        print(f"✓ Video saved in {save_time:.3f}s ({len(output_video_bytes) / 1024:.1f} KB)")

        # Total iteration time
        iteration_time = time.time() - iteration_start

        # Print detailed latency breakdown
        print(f"\n📊 Latency Breakdown (Iteration {i+1}):")
        print(f"  - Image loading:     {load_time:8.3f}s ({load_time/iteration_time*100:5.1f}%)")
        print(f"  - Remote inference:  {inference_time:8.3f}s ({inference_time/iteration_time*100:5.1f}%)")
        print(f"  - Video saving:      {save_time:8.3f}s ({save_time/iteration_time*100:5.1f}%)")
        print(f"  - Total:             {iteration_time:8.3f}s")

    print(f"\n✅ All {num_iterations} iteration(s) completed successfully!")
    print(f"📁 Final output saved to: {output_path}")
