"""
Kiểm tra màu sắc của ảnh nhiếp ảnh (photography).

Các vấn đề được phát hiện:
  - QUA_CHOI   : Màu quá chói (saturation cao + cast ấm mạnh)
  - QUA_NHAT   : Màu quá nhạt (saturation quá thấp)
  - QUA_CU     : Màu quá cũ/faded (contrast thấp + saturation trung bình thấp)
  - QUA_NONG   : Màu quá nóng (luminance cao + cast đỏ/cam mạnh)
  - QUA_LANH   : Màu quá lạnh (R/B ratio thấp, kênh xanh lam áp đảo)
  - HARMONIOUS : Màu sắc hài hòa

Calibration từ ảnh mẫu:
  Màu sắc hài hòa : lum=144.1  sat=41.8%  rb=1.512  contrast=56.1 → HARMONIOUS
  màu quá chói    : lum=132.0  sat=65.8%  rb=2.082  contrast=63.8 → QUA_CHOI
  màu quá nhạt    : lum=129.6  sat=13.7%  rb=1.068  contrast=54.1 → QUA_NHAT
  Màu sắc quá cũ  : lum=144.4  sat=35.1%  rb=1.473  contrast=40.7 → QUA_CU
  Màu quá nóng    : lum=167.5  sat=56.4%  rb=2.028  contrast=55.0 → QUA_NONG

Thứ tự ưu tiên trong quyết định:
  1. sat < 25%                          → QUA_NHAT
  2. rb_ratio < 0.9                     → QUA_LANH
  3. contrast < 45 và sat < 45%         → QUA_CU
  4. lum > 155 và rb_ratio > 1.8        → QUA_NONG
  5. sat > 55% và rb_ratio > 1.9        → QUA_CHOI
  6. còn lại                            → HARMONIOUS
"""

from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ngưỡng (có thể override khi gọi check_photo_color)
# ---------------------------------------------------------------------------
DEFAULT_SAT_TOO_PALE      = 25.0   # %  HSV sat dưới mức này → quá nhạt
DEFAULT_SAT_TOO_VIVID     = 55.0   # %  HSV sat trên mức này (kết hợp cast) → quá chói
DEFAULT_CONTRAST_OLD      = 45.0   # std luminance dưới mức này → nguy cơ quá cũ
DEFAULT_SAT_OLD           = 45.0   # %  sat dưới mức này kết hợp contrast thấp → quá cũ
DEFAULT_LUM_WARM          = 155.0  # luminance trung bình trên mức này → nguy cơ quá nóng
DEFAULT_RB_WARM           = 1.8    # tỷ lệ R/B trên mức này → cast ấm mạnh
DEFAULT_RB_VIVID          = 1.9    # tỷ lệ R/B trên mức này kết hợp sat cao → quá chói
DEFAULT_RB_COLD           = 0.9    # tỷ lệ R/B dưới mức này → cast lạnh/xanh lam


def _compute_metrics(arr: np.ndarray) -> dict:
    """Tính các chỉ số màu sắc từ mảng numpy uint8 (H×W×3)."""
    f = arr.astype(float)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]

    # Luminance theo Rec.601
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # HSV saturation
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat_v = np.where(cmax > 0, delta / cmax, 0.0)

    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    mean_lum  = float(lum.mean())
    blown_pct = float((lum > 240).mean() * 100)
    sat_mean  = float(sat_v.mean() * 100)
    contrast  = float(lum.std())
    rb_ratio  = float(mean_r / (mean_b + 1e-6))

    avg_rgb   = (mean_r + mean_g + mean_b) / 3.0
    r_cast    = float(mean_r - avg_rgb)
    g_cast    = float(mean_g - avg_rgb)
    b_cast    = float(mean_b - avg_rgb)

    return {
        "mean_luminance": round(mean_lum, 1),
        "blown_pct":      round(blown_pct, 2),
        "sat_hsv_pct":    round(sat_mean, 1),
        "contrast_std":   round(contrast, 1),
        "rb_ratio":       round(rb_ratio, 3),
        "r_cast":         round(r_cast, 1),
        "g_cast":         round(g_cast, 1),
        "b_cast":         round(b_cast, 1),
        "mean_r":         round(mean_r, 1),
        "mean_g":         round(mean_g, 1),
        "mean_b":         round(mean_b, 1),
    }


def _classify(
    m: dict,
    sat_too_pale:  float,
    sat_too_vivid: float,
    contrast_old:  float,
    sat_old:       float,
    lum_warm:      float,
    rb_warm:       float,
    rb_vivid:      float,
    rb_cold:       float,
) -> tuple[str, list[str]]:
    """
    Phân loại ảnh dựa trên các chỉ số đo.
    Trả về (status, danh_sach_ly_do).
    """
    reasons: list[str] = []
    sat  = m["sat_hsv_pct"]
    lum  = m["mean_luminance"]
    ctr  = m["contrast_std"]
    rb   = m["rb_ratio"]

    # 1. Quá nhạt
    if sat < sat_too_pale:
        reasons.append(f"saturation={sat:.1f}% < {sat_too_pale}% (quá nhạt/desaturated)")
        return "QUA_NHAT", reasons

    # 2. Quá lạnh (blue/cyan cast)
    if rb < rb_cold:
        reasons.append(f"R/B ratio={rb:.3f} < {rb_cold} (cast xanh lam/lạnh mạnh)")
        return "QUA_LANH", reasons

    # 3. Quá cũ / faded
    if ctr < contrast_old and sat < sat_old:
        reasons.append(f"contrast_std={ctr:.1f} < {contrast_old} (độ tương phản thấp)")
        reasons.append(f"saturation={sat:.1f}% < {sat_old}% (màu phai)")
        return "QUA_CU", reasons

    # 4. Quá nóng (warm cast + sáng)
    if lum > lum_warm and rb > rb_warm:
        reasons.append(f"luminance={lum:.1f} > {lum_warm} (ảnh sáng)")
        reasons.append(f"R/B ratio={rb:.3f} > {rb_warm} (cast đỏ/cam mạnh)")
        return "QUA_NONG", reasons

    # 5. Quá chói (saturation cao + warm cast)
    if sat > sat_too_vivid and rb > rb_vivid:
        reasons.append(f"saturation={sat:.1f}% > {sat_too_vivid}% (màu rực/chói)")
        reasons.append(f"R/B ratio={rb:.3f} > {rb_vivid} (cast ấm mạnh)")
        return "QUA_CHOI", reasons

    # 6. Hài hòa
    reasons.append("Các chỉ số luminance, saturation, contrast và color cast đều cân bằng")
    return "HARMONIOUS", reasons


def check_photo_color(
    image_path: str | Path,
    sat_too_pale:  float = DEFAULT_SAT_TOO_PALE,
    sat_too_vivid: float = DEFAULT_SAT_TOO_VIVID,
    contrast_old:  float = DEFAULT_CONTRAST_OLD,
    sat_old:       float = DEFAULT_SAT_OLD,
    lum_warm:      float = DEFAULT_LUM_WARM,
    rb_warm:       float = DEFAULT_RB_WARM,
    rb_vivid:      float = DEFAULT_RB_VIVID,
    rb_cold:       float = DEFAULT_RB_COLD,
) -> dict:
    """
    Kiểm tra màu sắc ảnh nhiếp ảnh.

    Args:
        image_path    : Đường dẫn tới file ảnh.
        sat_too_pale  : Ngưỡng saturation để coi là quá nhạt (%). Mặc định 25.
        sat_too_vivid : Ngưỡng saturation để coi là quá chói (%). Mặc định 55.
        contrast_old  : Ngưỡng contrast (std luminance) để coi là quá cũ. Mặc định 45.
        sat_old       : Ngưỡng saturation kết hợp để coi là quá cũ (%). Mặc định 45.
        lum_warm      : Ngưỡng luminance trung bình để coi là quá nóng. Mặc định 155.
        rb_warm       : Ngưỡng R/B ratio để coi là cast ấm. Mặc định 1.8.
        rb_vivid      : Ngưỡng R/B ratio để coi là quá chói. Mặc định 1.9.
        rb_cold       : Ngưỡng R/B ratio để coi là cast lạnh (xanh lam). Mặc định 0.9.

    Returns:
        dict gồm:
            - status  (str)  : "HARMONIOUS" | "QUA_CHOI" | "QUA_NHAT" | "QUA_CU" | "QUA_NONG" | "QUA_LANH"
            - is_ok   (bool) : True nếu status == "HARMONIOUS"
            - reasons (list) : Danh sách lý do phân loại
            - metrics (dict) : Các chỉ số đo lường chi tiết
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    metrics = _compute_metrics(arr)
    status, reasons = _classify(
        metrics,
        sat_too_pale, sat_too_vivid,
        contrast_old, sat_old,
        lum_warm, rb_warm, rb_vivid, rb_cold,
    )
    return {
        "status":  status,
        "is_ok":   status == "HARMONIOUS",
        "reasons": reasons,
        "metrics": metrics,
    }


# Nhãn hiển thị tiếng Việt
_STATUS_LABEL = {
    "HARMONIOUS": "Màu sắc hài hòa",
    "QUA_CHOI":   "Màu quá chói",
    "QUA_NHAT":   "Màu quá nhạt",
    "QUA_CU":     "Màu quá cũ",
    "QUA_NONG":   "Màu quá nóng",
    "QUA_LANH":   "Màu quá lạnh",
}


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        base = Path(__file__).parent
        paths = [
            base / "Màu sắc hài hòa.png",
            base / "màu quá chói.png",
            base / "màu quá nhạt.png",
            base / "Màu sắc quá cũ.png",
            base / "Màu quá nóng.png",
        ]

    for path in paths:
        result = check_photo_color(path)
        status = result["status"]
        label  = _STATUS_LABEL.get(status, status)
        icon   = "OK" if result["is_ok"] else "!!"
        m      = result["metrics"]

        print(f"\n[{icon}] {Path(path).name}")
        print(f"     Kết quả : {label} ({status})")
        print(f"     Lý do   : {' | '.join(result['reasons'])}")
        print(
            f"     Metrics : lum={m['mean_luminance']}  sat={m['sat_hsv_pct']}%"
            f"  contrast={m['contrast_std']}  R/B={m['rb_ratio']}"
        )
