import asyncio
import os
import time
import uuid
import numpy as np
from google import genai
from google.genai import types

# ================= CONFIG =================
API_KEY = "AQ.Ab8RN6Lcv9M_zUHXF-F5oDRDBDSxCGB-Q3ccu6yeoQ43Yp74Cg"
MODEL = "gemini-2.5-flash-image"

IMAGE_PATH = "test_vision_model/sample1.png"
OUTPUT_DIR = "test_vision_model/output/test_gemini_flex"

BASE_PROMPT = """
Edit the image: make the man wear sunglasses and a long sleeve shirt.
Keep face, pose, and background unchanged.
"""

TOTAL_REQUESTS = 10
CCU = 10
WARMUP_REQUESTS = 3

TIMEOUT_MS = 120000

# ================= LOAD IMAGE =================
with open(IMAGE_PATH, "rb") as f:
    IMAGE_BYTES = f.read()

# ================= INIT CLIENT =================
client = genai.Client(api_key=API_KEY)

# ================= SINGLE CALL =================
def call_api_sync(request_idx):
    start = time.time()

    timestamp = int(time.time() * 1000)
    request_id = str(uuid.uuid4())

    dynamic_prompt = f"""
    {BASE_PROMPT}
    [request_id: {request_id}]
    [timestamp: {timestamp}]
    [req_index: {request_idx}]
    """

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=dynamic_prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=IMAGE_BYTES
                            )
                        )
                    ]
                )
            ],
            config={
                "service_tier": "flex",
                "http_options": {"timeout": TIMEOUT_MS}
            },
        )

        # ===== VALIDATION & SAVE =====
        is_valid = False

        if response and hasattr(response, "candidates"):
            for cand in response.candidates:
                if hasattr(cand, "content") and cand.content:
                    for part in cand.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            is_valid = True
                            out_path = os.path.join(OUTPUT_DIR, f"req_{request_idx:03d}.png")
                            with open(out_path, "wb") as f:
                                f.write(part.inline_data.data)
                            print(f"[SAVED] req {request_idx} → {out_path}")
                            break

        latency = time.time() - start

        if not is_valid:
            print(f"[INVALID] req {request_idx}: no image in response")

        return latency, is_valid

    except Exception as e:
        latency = time.time() - start
        print(f"[ERROR] req {request_idx}: {e}")
        return latency, False

# ================= BENCHMARK =================
async def worker(i, semaphore, latencies):
    async with semaphore:
        loop = asyncio.get_event_loop()
        latency, success = await loop.run_in_executor(
            None, call_api_sync, i
        )
        latencies.append((latency, success))

async def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    semaphore = asyncio.Semaphore(CCU)
    latencies = []

    tasks = [worker(i, semaphore, latencies) for i in range(TOTAL_REQUESTS)]

    start_all = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - start_all

    return latencies, total_time

# ================= MAIN =================
async def main():

    latencies, total_time = await run_benchmark()

    if len(latencies) == 0:
        print("No data collected.")
        return

    success = sum(1 for _, ok in latencies if ok)
    fail = sum(1 for _, ok in latencies if not ok)
    latencies_sorted = sorted(lat for lat, _ in latencies)

    p50 = np.percentile(latencies_sorted, 50)
    p90 = np.percentile(latencies_sorted, 90)
    p95 = np.percentile(latencies_sorted, 95)

    throughput = success / total_time
    error_rate = fail / (success + fail)

    print("\n========== RESULT ==========")
    print(f"CCU: {CCU}")
    print(f"Total Requests: {TOTAL_REQUESTS}")
    print(f"Success: {success}")
    print(f"Fail: {fail}")
    print(f"Error Rate: {error_rate * 100:.2f}%")

    print(f"\nThroughput: {throughput:.2f} req/s")

    print("\nLatency:")
    print(f"P50: {p50:.2f}s")
    print(f"P90: {p90:.2f}s")
    print(f"P95: {p95:.2f}s")

    print("============================")


# ================= RUN =================
if __name__ == "__main__":
    asyncio.run(main())
