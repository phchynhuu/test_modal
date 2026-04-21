"""
Kiểm tra kích thước ảnh:
  - check_size : ảnh phải có kích thước tối thiểu 1200×1200 px
"""

from pathlib import Path
from PIL import Image


def check_size(
    image_path: str | Path,
    min_width: int = 1200,
    min_height: int = 1200,
) -> dict:
    """
    Kiểm tra ảnh có đạt kích thước tối thiểu hay không.

    Args:
        image_path: Đường dẫn tới file ảnh.
        min_width: Chiều rộng tối thiểu (pixel). Mặc định 1200.
        min_height: Chiều cao tối thiểu (pixel). Mặc định 1200.

    Returns:
        dict gồm:
            - passed (bool): True nếu ảnh đạt kích thước yêu cầu
            - width (int): Chiều rộng thực tế (pixel)
            - height (int): Chiều cao thực tế (pixel)
            - min_width (int): Chiều rộng tối thiểu yêu cầu
            - min_height (int): Chiều cao tối thiểu yêu cầu
            - errors (list[str]): Danh sách lỗi (nếu có)
    """
    img = Image.open(image_path)
    width, height = img.size

    errors: list[str] = []

    if width < min_width:
        errors.append(
            f"Chiều rộng ({width}px) nhỏ hơn yêu cầu ({min_width}px)"
        )

    if height < min_height:
        errors.append(
            f"Chiều cao ({height}px) nhỏ hơn yêu cầu ({min_height}px)"
        )

    return {
        "passed": len(errors) == 0,
        "width": width,
        "height": height,
        "min_width": min_width,
        "min_height": min_height,
        "errors": errors,
    }


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        # Tìm tất cả ảnh trong thư mục hiện tại
        base = Path(__file__).parent
        paths = sorted(
            p
            for p in base.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
        )

    if not paths:
        print("Không tìm thấy ảnh nào. Truyền đường dẫn ảnh qua argument.")
        sys.exit(1)

    for path in paths:
        result = check_size(path)
        status = "OK" if result["passed"] else "FAIL"
        print(f"[{status}] {Path(path).name}")
        print(f"       kích thước: {result['width']}×{result['height']} px")
        if result["errors"]:
            for err in result["errors"]:
                print(f"       ✗ {err}")
        print()
