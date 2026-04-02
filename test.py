from openai import OpenAI
import time

inference_start = time.time()

client = OpenAI(
    base_url="https://dorathekid110--sglang-qwen35-serve-dev.modal.run",
    api_key="no-key-needed",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Hello!"}],
)


inference_time = time.time() - inference_start
print(f"✓ Inference completed in {inference_time:.3f}s")
