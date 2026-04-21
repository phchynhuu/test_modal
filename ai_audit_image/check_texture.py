"""
Kiểm tra mức độ scale của texture hatching có nằm trong khoảng 40–60%.

Calibration từ sample images:
  - spacing ~9.6px  ≈ 50% scale  →  reference_period = 19.25px
  - Quá nhỏ (<40%): spacing < 7.7px
  - Chuẩn (40–60%): spacing 7.7–11.55px
  - Quá to (>60%):  spacing > 11.55px
"""

from pathlib import Path
import numpy as np
from PIL import Image


def _content_bbox(arr: np.ndarray):
    """Trả về (top, bottom, left, right) và mask content."""
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
    top = int(np.argmax(rows))
    bottom = int(len(rows) - 1 - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return top, bottom, left, right, mask


def _line_spacing(values: np.ndarray, dark_threshold: int = 230) -> float | None:
    """
    Đo khoảng cách trung bình giữa các run pixel tối trong 1 dòng/cột.
    Trả về median spacing (px), hoặc None nếu không đủ data.
    """
    is_dark = values < dark_threshold
    in_run = False
    run_starts: list[int] = []
    for i, d in enumerate(is_dark.tolist()):
        if d and not in_run:
            run_starts.append(i)
            in_run = True
        elif not d:
            in_run = False
    if len(run_starts) < 2:
        return None
    spacings = np.diff(run_starts)
    # Lọc outlier: spacing quá nhỏ < 2px hoặc quá lớn > 60px
    spacings = spacings[(spacings >= 2) & (spacings <= 60)]
    return float(np.median(spacings)) if len(spacings) >= 1 else None


def check_texture_scale(
    image_path: str | Path,
    min_scale: float = 40.0,
    max_scale: float = 60.0,
    reference_period_px: float = 19.25,
    n_scan_lines: int = 60,
) -> dict:
    """
    Kiểm tra mức độ scale của texture hatching có nằm trong khoảng hợp lệ.

    Scale được tính bằng: (spacing_px / reference_period_px) × 100 (%)
    Calibration từ sample: spacing ~9.6px ≈ 50% scale → reference = 19.25px.

    Args:
        image_path          : Đường dẫn tới file ảnh.
        min_scale           : Ngưỡng dưới (%). Mặc định 40.
        max_scale           : Ngưỡng trên (%). Mặc định 60.
        reference_period_px : Period (px) tương ứng 100% scale. Mặc định 19.25.
        n_scan_lines        : Số dòng/cột scan qua content. Mặc định 60.

    Returns:
        dict gồm:
            - is_valid   (bool)       : True nếu scale trong [min_scale, max_scale]
            - scale_pct  (float|None) : scale đo được (%)
            - spacing_px (float|None) : khoảng cách trung bình giữa texture lines (px)
            - n_samples  (int)        : số measurement hợp lệ thu được
            - status     (str)        : "OK" | "TOO_FINE" | "TOO_COARSE" | "INSUFFICIENT_DATA"
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    t, b, l, r, mask = _content_bbox(arr)

    gray = np.array(Image.open(image_path).convert("L"))

    h_content = b - t
    w_content = r - l
    spacings: list[float] = []

    # Scan ngang
    ys = np.linspace(t + h_content * 0.15, t + h_content * 0.85, n_scan_lines).astype(int)
    for y in ys:
        if y >= gray.shape[0]:
            continue
        row_mask = mask[y, l:r]
        if row_mask.mean() < 0.25:  # bỏ dòng phần lớn là nền
            continue
        s = _line_spacing(gray[y, l:r])
        if s is not None:
            spacings.append(s)

    # Scan dọc
    xs = np.linspace(l + w_content * 0.15, l + w_content * 0.85, n_scan_lines).astype(int)
    for x in xs:
        if x >= gray.shape[1]:
            continue
        col_mask = mask[t:b, x]
        if col_mask.mean() < 0.25:
            continue
        s = _line_spacing(gray[t:b, x])
        if s is not None:
            spacings.append(s)

    if len(spacings) < 5:
        return {
            "is_valid": False,
            "scale_pct": None,
            "spacing_px": None,
            "n_samples": len(spacings),
            "status": "INSUFFICIENT_DATA",
        }

    spacing_px = float(np.median(spacings))
    scale_pct = (spacing_px / reference_period_px) * 100.0

    if scale_pct < min_scale:
        status = "TOO_FINE"
    elif scale_pct > max_scale:
        status = "TOO_COARSE"
    else:
        status = "OK"

    return {
        "is_valid": status == "OK",
        "scale_pct": round(scale_pct, 1),
        "spacing_px": round(spacing_px, 2),
        "n_samples": len(spacings),
        "status": status,
    }


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        base = Path(__file__).parent
        paths = [
            base / "texture chuẩn.png",
            base / "texture chuẩn 02.png",
            base / "Texture quá nhỏ.png",
            base / "Texture quá nhỏ 01.png",
            base / "Texture quá to.png",
            base / "Texture quá to.01png.png",
        ]

    for path in paths:
        result = check_texture_scale(path)
        status = result["status"]
        print(f"[{status:<17}] {Path(path).name}")
        print(f"       scale={result['scale_pct']}%  spacing={result['spacing_px']}px  samples={result['n_samples']}")
        print()
