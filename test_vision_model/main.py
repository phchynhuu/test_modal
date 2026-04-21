import requests
import base64
import numpy as np
from PIL import Image

# === Config ===
API_URL = "https://llm.mservice.io/v1/chat/completions"
API_KEY = "sk-123123"
IMAGE_PATH = "banner.jpg"

# === Encode local image ===
with open(IMAGE_PATH, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")


def validate_imagesize(
    expected_width: int = 1920,
    expected_height: int = 1080,
    image_path: str = IMAGE_PATH,
) -> dict:
    """Validate that the image matches the expected pixel dimensions exactly.

    Returns a dict with actual dimensions on success, raises ValueError on failure.
    """
    with Image.open(image_path) as img:
        width, height = img.size

    if width != expected_width or height != expected_height:
        raise ValueError(
            f"Image size {width}x{height}px does not match expected {expected_width}x{expected_height}px"
        )

    return {"width": width, "height": height}


def validate_image_color(
    image_path: str = IMAGE_PATH,
    min_brightness: float = 30.0,
    max_brightness: float = 220.0,
    min_saturation: float = 10.0,
) -> dict:
    """Validate image color properties: brightness and color saturation.

    Brightness is the mean pixel value across all channels (0–255).
    Saturation is the mean of (max_channel - min_channel) per pixel (0–255);
    0 = pure grayscale, 255 = fully saturated.

    Returns a dict with color metrics on success, raises ValueError on failure.
    """
    with Image.open(image_path) as img:
        pixels = np.array(img.convert("RGB"), dtype=np.float32)

    brightness = float(pixels.mean())
    if brightness < min_brightness:
        raise ValueError(
            f"Image is too dark (brightness={brightness:.1f}, min={min_brightness})"
        )
    if brightness > max_brightness:
        raise ValueError(
            f"Image is too bright/washed out (brightness={brightness:.1f}, max={max_brightness})"
        )

    # Saturation proxy: per-pixel spread across R, G, B channels
    saturation = float((pixels.max(axis=2) - pixels.min(axis=2)).mean())
    if saturation < min_saturation:
        raise ValueError(
            f"Image appears desaturated/near-grayscale (saturation={saturation:.1f}, min={min_saturation})"
        )

    return {
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
    }


# === Tool definitions ===
tools = [
    {
        "type": "function",
        "function": {
            "name": "validate_imagesize",
            "description": "Check that the image matches the expected pixel dimensions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expected_width":  {"type": "integer", "description": "Expected image width in pixels"},
                    "expected_height": {"type": "integer", "description": "Expected image height in pixels"},
                },
                "required": ["expected_width", "expected_height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_image_color",
            "description": "Check image brightness and color saturation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_brightness": {"type": "number", "description": "Minimum acceptable brightness (0-255)"},
                    "max_brightness": {"type": "number", "description": "Maximum acceptable brightness (0-255)"},
                    "min_saturation": {"type": "number", "description": "Minimum acceptable saturation (0-255)"},
                },
                "required": [],
            },
        },
    },
]

# === Tool dispatcher ===
import json

def dispatch_tool(name: str, args: dict) -> str:
    try:
        if name == "validate_imagesize":
            result = validate_imagesize(**args)
        elif name == "validate_image_color":
            result = validate_image_color(**args)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except ValueError as e:
        result = {"error": str(e)}
    return json.dumps(result)


# === Agentic loop ===
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

messages = [
    {
        "role": "system",
        "content": "You are an expert image quality auditor. Use the provided tools to validate the image, then return a final audit summary as valid JSON.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Audit this banner image and return ONLY valid JSON."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ],
    },
]

while True:
    payload = {
        "model": "gpt-4o",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "tools": tools,
        "messages": messages,
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    print("Status:", response.status_code)

    data = response.json()
    assistant_msg = data["choices"][0]["message"]
    messages.append(assistant_msg)

    # No more tool calls — done
    if assistant_msg.get("tool_calls") is None:
        print(assistant_msg.get("content"))
        break

    # Execute each tool call and feed results back
    for tool_call in assistant_msg["tool_calls"]:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        print(f"Tool call: {name}({args})")

        result = dispatch_tool(name, args)
        print(f"  -> {result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": result,
        })
