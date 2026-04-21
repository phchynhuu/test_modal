"""
Kiểm tra màu sắc của ảnh minh họa có nằm trong khoảng saturation hợp lệ.

Metric: median HSL saturation của vùng fill (loại bỏ outline và nền)
  - Quá nhạt : sat_hsl < 75%   (màu nhợt, desaturated)
  - Đúng      : sat_hsl ∈ [75, 98]%
  - Quá tươi  : sat_hsl > 98%   (màu sặc sỡ, fully saturated)

Calibration data:
  Màu sắc đúng.png       sat=90.0%  → OK
  Màu sắc đúng 01.png    sat=84.7%  → OK
  Màu quá nhạt.png       sat=61.2%  → TOO_PALE
  Màu quá nhạt 01.png    sat=59.1%  → TOO_PALE
  Màu quá tươi.png       sat=100%   → TOO_VIVID
  Màu quá tươi 01.png    sat=100%   → TOO_VIVID
"""

from pathlib import Path
import numpy as np
from PIL import Image


def _content_mask(arr: np.ndarray) -> np.ndarray:
    if arr.shape[2] == 4:
        return arr[:, :, 3] > 10
    rgb = arr[:, :, :3]
    corners = [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]
    bg = np.median(corners, axis=0)
    diff = np.abs(rgb.astype(int) - bg.astype(int)).max(axis=2)
    return diff > 20


def _hsl_saturation(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tính HSL saturation (0–1) cho mảng pixel đã normalize về [0,1]."""
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    lightness = (cmax + cmin) / 2.0
    denom = 1.0 - np.abs(2.0 * lightness - 1.0)
    sat = np.where(denom > 0.01, delta / denom, 0.0)
    return np.clip(sat, 0.0, 1.0)


def check_color(
    image_path: str | Path,
    min_saturation: float = 75.0,
    max_saturation: float = 98.0,
) -> dict:
    """
    Kiểm tra saturation màu sắc có nằm trong khoảng hợp lệ.

    Args:
        image_path      : Đường dẫn tới file ảnh.
        min_saturation  : Ngưỡng dưới HSL saturation (%). Mặc định 75.
        max_saturation  : Ngưỡng trên HSL saturation (%). Mặc định 98.

    Returns:
        dict gồm:
            - is_valid        (bool)       : True nếu saturation trong khoảng hợp lệ
            - saturation_pct  (float|None) : median HSL saturation của fill pixels (%)
            - status          (str)        : "OK" | "TOO_PALE" | "TOO_VIVID" | "INSUFFICIENT_DATA"
            - n_fill_pixels   (int)        : số pixel fill được dùng để đo
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    mask = _content_mask(arr)

    rgb = arr[:, :, :3].astype(float) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Giữ lại vùng fill: content, loại outline cứng (<25% gray) và gần trắng (>97%)
    fill_mask = mask & (gray > 0.25) & (gray < 0.97)

    n_fill = int(fill_mask.sum())
    if n_fill < 200:
        return {
            "is_valid": False,
            "saturation_pct": None,
            "status": "INSUFFICIENT_DATA",
            "n_fill_pixels": n_fill,
        }

    r = rgb[:, :, 0][fill_mask]
    g = rgb[:, :, 1][fill_mask]
    b = rgb[:, :, 2][fill_mask]

    sat = _hsl_saturation(r, g, b)
    sat_median = float(np.median(sat)) * 100.0

    if sat_median < min_saturation:
        status = "TOO_PALE"
    elif sat_median > max_saturation:
        status = "TOO_VIVID"
    else:
        status = "OK"

    return {
        "is_valid": status == "OK",
        "saturation_pct": round(sat_median, 1),
        "status": status,
        "n_fill_pixels": n_fill,
    }


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        base = Path(__file__).parent
        paths = [
            base / "Màu sắc đúng.png",
            base / "Màu sắc đúng 01.png",
            base / "Màu quá nhạt.png",
            base / "Màu quá nhạt 01.png",
            base / "Màu quá tươi.png",
            base / "Màu quá tươi 01.png",
        ]

    for path in paths:
        result = check_color(path)
        status = result["status"]
        print(f"[{status:<17}] {Path(path).name}")
        print(f"       saturation={result['saturation_pct']}%  fill_pixels={result['n_fill_pixels']}")
        print()
