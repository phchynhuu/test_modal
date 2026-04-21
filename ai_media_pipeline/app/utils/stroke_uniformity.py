"""
Stroke Uniformity Checker

Kiểm tra ảnh illustration có stroke (đường viền) đều hay không.

Thuật toán:
  1. Convert ảnh sang grayscale, loại bỏ background đen (flood fill từ góc ảnh)
  2. Threshold pixel tối trong subject → binary stroke mask
  3. Distance Transform: mỗi pixel foreground biết khoảng cách đến edge gần nhất
  4. Zhang-Suen Thinning → skeleton (centerline) 1px
  5. Chỉ lấy "body point" (đúng 2 neighbors) → loại junction / endpoint artifact
  6. Stroke width = 2 × dist_transform tại mỗi body point
  7. Hai tiêu chí kết luận:
       - CV (std/mean của outline stroke 5–20px) ≥ threshold → không đều
       - Thin ratio (% stroke body < 4px) ≥ threshold → không đều

Dependencies: opencv-contrib-python, numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
from pathlib import Path

import cv2
import numpy as np


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StrokeUniformityResult:
    is_uniform: bool
    cv: float                   # Coefficient of Variation (std / mean) của outline stroke
    thin_stroke_ratio: float    # Tỉ lệ điểm stroke mỏng bất thường (< 5px) / tổng
    mean_width_px: float        # Stroke width trung bình (pixels)
    std_width_px: float         # Độ lệch chuẩn stroke width
    sample_count: int           # Số điểm đo được
    cv_threshold: float
    thin_ratio_threshold: float

    def summary(self) -> str:
        verdict = "ĐỒNG ĐỀU" if self.is_uniform else "KHÔNG ĐỀU"
        return (
            f"[{verdict}] CV={self.cv:.3f} | thin_ratio={self.thin_stroke_ratio:.1%} | "
            f"mean={self.mean_width_px:.1f}px ± {self.std_width_px:.1f}px | "
            f"samples={self.sample_count}"
        )


# ── Core function ─────────────────────────────────────────────────────────────

ImageInput = Union[str, Path, bytes, np.ndarray]


def check_stroke_uniformity(
    image: ImageInput,
    stroke_dark_threshold: int = 80,
    min_stroke_area: int = 50,
    cv_threshold: float = 0.55,
    thin_ratio_threshold: float = 0.08,
) -> StrokeUniformityResult:
    """
    Kiểm tra độ đồng đều của stroke trong ảnh.

    Sử dụng 2 tiêu chí kết hợp:
      1. CV (Coefficient of Variation): phân phối độ rộng stroke đồng đều không
      2. Thin stroke ratio: tỉ lệ điểm stroke mỏng bất thường (dấu hiệu stroke lỗi)

    Kết luận KHÔNG ĐỀU nếu THỎA MÃN BẤT KỲ 1 điều kiện:
      - CV >= cv_threshold
      - thin_stroke_ratio >= thin_ratio_threshold

    Args:
        image:
            Đường dẫn file, bytes raw, hoặc numpy array (BGR/BGRA/grayscale).
        stroke_dark_threshold:
            Grayscale value <= ngưỡng này được coi là stroke (mặc định 80).
        min_stroke_area:
            Lọc noise: bỏ connected component nhỏ hơn số pixel này.
        cv_threshold:
            Ngưỡng CV. Mặc định 0.55 (thực nghiệm trên illustration style).
        thin_ratio_threshold:
            Ngưỡng thin ratio. Mặc định 0.08 (>8% stroke mỏng → không đều).

    Returns:
        StrokeUniformityResult với các metric chi tiết.

    Raises:
        ValueError: Nếu không load được ảnh hoặc không tìm thấy stroke nào.
    """
    gray = _load_as_grayscale(image)
    stroke_mask = _extract_stroke_mask(gray, stroke_dark_threshold, min_stroke_area)

    if stroke_mask.sum() == 0:
        raise ValueError(
            "Không tìm thấy vùng stroke nào. "
            "Thử tăng stroke_dark_threshold hoặc kiểm tra ảnh đầu vào."
        )

    widths_range, widths_outline = _measure_stroke_widths(stroke_mask)

    if len(widths_outline) < 10:
        raise ValueError(
            f"Chỉ đo được {len(widths_outline)} outline stroke points — quá ít. "
            "Thử giảm min_stroke_area hoặc kiểm tra stroke_dark_threshold."
        )

    mean_w = float(np.mean(widths_outline))
    std_w = float(np.std(widths_outline))
    cv = std_w / mean_w if mean_w > 0 else float("inf")

    # Thin stroke ratio: tỉ lệ điểm width < 4px trong tổng 2–20px
    # Stroke đều: hầu hết điểm tập trung quanh width trung bình, ít điểm quá mỏng
    # Stroke không đều: nhiều điểm mỏng bất thường (đường viền bị nhỏ lại)
    thin_in_range = widths_range[widths_range < 4.0]
    thin_ratio = (
        float(len(thin_in_range) / len(widths_range)) if len(widths_range) > 0 else 0.0
    )

    is_uniform = (cv < cv_threshold) and (thin_ratio < thin_ratio_threshold)

    return StrokeUniformityResult(
        is_uniform=is_uniform,
        cv=round(cv, 4),
        thin_stroke_ratio=round(thin_ratio, 4),
        mean_width_px=round(mean_w, 2),
        std_width_px=round(std_w, 2),
        sample_count=len(widths_outline),
        cv_threshold=cv_threshold,
        thin_ratio_threshold=thin_ratio_threshold,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_as_grayscale(image: ImageInput) -> np.ndarray:
    """Load ảnh từ nhiều dạng input và trả về grayscale numpy array."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:             # BGRA
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if isinstance(image, (str, Path)):
        arr = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Không đọc được file ảnh: {image}")
        return _load_as_grayscale(arr)

    if isinstance(image, bytes):
        arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("Không decode được bytes ảnh.")
        return _load_as_grayscale(arr)

    raise TypeError(f"Không hỗ trợ kiểu đầu vào: {type(image)}")


def _remove_background(gray: np.ndarray) -> np.ndarray:
    """
    Tạo mask cho vùng background (đen thuần) bằng flood fill từ 4 góc ảnh.

    Illustration thường có nền đen liên thông với các góc ảnh.
    Trả về mask 255 = background, 0 = subject.
    """
    h, w = gray.shape

    # Pixel rất tối (< 20) là ứng viên background
    _, very_dark = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

    temp = very_dark.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill từ 4 góc để lấy vùng background liên thông
    for (fy, fx) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if temp[fy, fx] > 0:
            cv2.floodFill(temp, flood_mask, (fx, fy), 128)

    # Vùng đã fill = background
    background_mask = (flood_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
    return background_mask


def _extract_stroke_mask(
    gray: np.ndarray,
    dark_threshold: int,
    min_area: int,
) -> np.ndarray:
    """
    Tạo binary mask cho vùng stroke.

    Stroke là pixel tối nằm trong subject (không phải background đen).
    """
    # Loại bỏ background đen liên thông từ góc ảnh
    bg_mask = _remove_background(gray)

    # Pixel tối trong subject = stroke
    _, dark = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    stroke_candidates = cv2.bitwise_and(dark, cv2.bitwise_not(bg_mask))

    # Lọc noise: giữ lại connected component đủ lớn
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        stroke_candidates, connectivity=8
    )
    cleaned = np.zeros_like(stroke_candidates)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_idx] = 255

    return cleaned


def _measure_stroke_widths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Đo độ dày stroke tại các điểm "body" trên skeleton (nc == 3).

    "Body point": điểm skeleton có đúng 2 neighbors → không phải junction hay endpoint
    → Loại bỏ artifact tại giao điểm (junction) và đầu stroke (endpoint).

    Returns:
        widths_range:   width tại body points trong 2–20px (dùng để tính thin_ratio)
        widths_outline: width tại body points trong 5–20px (dùng để tính CV)
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_5)
    skeleton = _skeletonize(mask)

    skel_binary = (skeleton > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(
        skel_binary.astype(np.float32), -1, np.ones((3, 3), np.float32)
    )
    # nc == 3: bản thân + đúng 2 neighbors → body point
    body_ys, body_xs = np.where((skel_binary > 0) & (neighbor_count == 3))

    if len(body_ys) == 0:
        return np.array([]), np.array([])

    raw_widths = dist[body_ys, body_xs] * 2.0

    widths_range   = raw_widths[(raw_widths >= 2.0) & (raw_widths <= 20.0)]
    widths_outline = raw_widths[(raw_widths >= 5.0) & (raw_widths <= 20.0)]

    return widths_range, widths_outline


def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """
    Thinning morphological để ra centerline (skeleton) 1px.
    Dùng thuật toán Zhang-Suen tích hợp trong OpenCV.
    """
    thinned = cv2.ximgproc.thinning(
        binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
    )
    return thinned
