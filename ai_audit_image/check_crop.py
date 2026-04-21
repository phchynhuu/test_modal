"""
Kiểm tra chất lượng ảnh minh họa:
  - check_crop_tight : crop sát mép (margin ≤ 100px mỗi cạnh)
  - check_texture_scale : scale texture hatching trong khoảng 40–60%
"""

from pathlib import Path
import numpy as np
from PIL import Image


def check_crop_tight(image_path: str | Path, max_margin_px: int = 100) -> dict:
    """
    Kiểm tra ảnh có được crop sát mép hay không.

    Args:
        image_path: Đường dẫn tới file ảnh.
        max_margin_px: Ngưỡng tối đa cho phép (pixel). Mặc định 100px.

    Returns:
        dict gồm:
            - is_tight (bool): True nếu crop đạt yêu cầu
            - margins (dict): số pixel trống mỗi cạnh {top, bottom, left, right}
            - exceeds (list[str]): các cạnh vượt ngưỡng
    """
    img = Image.open(image_path)

    if img.mode in ("RGBA", "LA"):
        # Dùng kênh alpha để tìm vùng có nội dung
        alpha = np.array(img.getchannel("A"))
        mask = alpha > 10  # pixel gần trong suốt coi là nền
    else:
        # Phát hiện màu nền từ 4 góc ảnh
        rgb = np.array(img.convert("RGB"))
        h, w = rgb.shape[:2]
        corners = [
            rgb[0, 0],
            rgb[0, w - 1],
            rgb[h - 1, 0],
            rgb[h - 1, w - 1],
        ]
        bg_color = np.median(corners, axis=0).astype(np.uint8)

        # Mask: pixel khác màu nền quá tolerance = 20
        diff = np.abs(rgb.astype(int) - bg_color.astype(int))
        mask = diff.max(axis=2) > 20

    rows = np.any(mask, axis=1)  # True ở row nào có nội dung
    cols = np.any(mask, axis=0)  # True ở col nào có nội dung

    if not rows.any():
        return {
            "is_tight": False,
            "margins": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "exceeds": ["image appears empty"],
        }

    top = int(np.argmax(rows))
    bottom = int(len(rows) - 1 - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - 1 - np.argmax(cols[::-1]))

    h, w = mask.shape
    margins = {
        "top": top,
        "bottom": h - 1 - bottom,
        "left": left,
        "right": w - 1 - right,
    }

    exceeds = [side for side, px in margins.items() if px > max_margin_px]

    return {
        "is_tight": len(exceeds) == 0,
        "margins": margins,
        "exceeds": exceeds,
    }


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        # Chạy với ảnh sample có sẵn
        base = Path(__file__).parent
        paths = [
            base / "Crop đúng.png",
            base / "Crop đúng 01.png",
            base / "Chưa crop sát mép.png",
            base / "Chưa crop sát mép 01.png",
        ]

    for path in paths:
        result = check_crop_tight(path)
        status = "OK" if result["is_tight"] else "FAIL"
        print(f"[{status}] {Path(path).name}")
        print(f"       margins: {result['margins']}")
        if result["exceeds"]:
            print(f"       vượt 100px: {result['exceeds']}")
        print()
