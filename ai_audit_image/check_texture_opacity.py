"""
Kiểm tra texture opacity có nằm trong khoảng hợp lệ.

Metric: hp_ratio = mean(|gray - blurred|) / content_range × 100  (%)
  - Đo tỉ lệ năng lượng texture (high-pass) so với dynamic range của content
  - Chuẩn (calibrated từ sample): hp_ratio ∈ [17, 25]
  - Quá nhạt: hp_ratio < 17  (texture lines mờ, ít tương phản)
  - Quá đậm:  hp_ratio > 25  (texture lines đậm, quá tương phản)

Calibration data:
  texture chuẩn.png      hp=19.57  → OK
  texture chuẩn 01.png   hp=19.00  → OK
  quá nhạt.png           hp= 6.90  → TOO_LIGHT
  quá nhạt 01.png        hp=16.57  → TOO_LIGHT
  quá đậm.png            hp=35.50  → TOO_DARK
  quá đậm 01.png         hp=25.70  → TOO_DARK
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


def _content_bbox_and_mask(arr: np.ndarray):
    """Trả về (top, bottom, left, right, mask) của vùng content."""
    if arr.shape[2] == 4:
        mask = arr[:, :, 3] > 10
    else:
        rgb = arr[:, :, :3]
        corners = [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]
        bg = np.median(corners, axis=0)
        diff = np.abs(rgb.astype(int) - bg.astype(int)).max(axis=2)
        mask = diff > 20

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top    = int(np.argmax(rows))
    bottom = int(len(rows) - 1 - np.argmax(rows[::-1]))
    left   = int(np.argmax(cols))
    right  = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return top, bottom, left, right, mask


def check_texture_opacity(
    image_path: str | Path,
    min_hp_ratio: float = 17.0,
    max_hp_ratio: float = 25.0,
) -> dict:
    """
    Kiểm tra texture opacity của ảnh có nằm trong khoảng hợp lệ.

    Args:
        image_path   : Đường dẫn tới file ảnh.
        min_hp_ratio : Ngưỡng dưới của hp_ratio (%). Mặc định 17.
        max_hp_ratio : Ngưỡng trên của hp_ratio (%). Mặc định 25.

    Returns:
        dict gồm:
            - is_valid   (bool)       : True nếu opacity trong khoảng hợp lệ
            - hp_ratio   (float|None) : giá trị đo được (%)
            - status     (str)        : "OK" | "TOO_LIGHT" | "TOO_DARK" | "INSUFFICIENT_DATA"
    """
    pil_img  = Image.open(image_path)
    arr      = np.array(pil_img.convert("RGBA"))
    t, b, l, r, mask = _content_bbox_and_mask(arr)

    pil_gray = pil_img.convert("L")
    gray     = np.array(pil_gray, dtype=float)

    content_size = min(b - t, r - l)
    if content_size < 50:
        return {"is_valid": False, "hp_ratio": None, "status": "INSUFFICIENT_DATA"}

    # Blur radius scale theo content size để nhất quán giữa các ảnh khác kích thước
    blur_radius = max(3, int(content_size * 0.008))

    blurred  = np.array(pil_gray.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=float)
    texture  = gray - blurred   # high-pass: dương = sáng hơn base, âm = tối hơn base (hatching)

    # Vùng fill hợp lệ: bỏ outline cứng (<70) và vùng gần trắng (>250)
    fill_mask = mask & (gray > 70) & (gray < 250)

    if fill_mask.sum() < 200:
        return {"is_valid": False, "hp_ratio": None, "status": "INSUFFICIENT_DATA"}

    fill_gray    = gray[fill_mask]
    fill_texture = texture[fill_mask]

    content_range = float(np.percentile(fill_gray, 95) - np.percentile(fill_gray, 5))
    if content_range < 1:
        return {"is_valid": False, "hp_ratio": None, "status": "INSUFFICIENT_DATA"}

    mean_abs_texture = float(np.mean(np.abs(fill_texture)))
    hp_ratio = (mean_abs_texture / content_range) * 100.0

    if hp_ratio < min_hp_ratio:
        status = "TOO_LIGHT"
    elif hp_ratio > max_hp_ratio:
        status = "TOO_DARK"
    else:
        status = "OK"

    return {
        "is_valid": status == "OK",
        "hp_ratio": round(hp_ratio, 2),
        "status": status,
    }


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        base = Path(__file__).parent
        paths = [
            base / "Texture opacity chuẩn.png",
            base / "Texture opacity chuẩn 01.png",
            base / "Texture opacity quá nhạt.png",
            base / "Texture opacity quá nhạt 01.png",
            base / "Texture opacity quá đậm.png",
            base / "Texture opacity quá đậm 01.png",
        ]

    for path in paths:
        result = check_texture_opacity(path)
        status = result["status"]
        print(f"[{status:<12}] {Path(path).name}")
        print(f"             hp_ratio={result['hp_ratio']}%")
        print()
