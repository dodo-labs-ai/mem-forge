"""Deterministic upright PNG regions, without source or model semantics."""

from io import BytesIO
from math import ceil
from hashlib import sha256

from PIL import Image


def pixel_digest(image: Image.Image) -> str:
    return sha256(f"{image.mode}:{image.width}:{image.height}:".encode() + image.tobytes()).hexdigest()


def open_upright_png(body: bytes) -> Image.Image:
    with Image.open(BytesIO(body)) as image:
        image.verify()
    with Image.open(BytesIO(body)) as image:
        if (image.format != "PNG" or getattr(image, "n_frames", 1) != 1
                or image.getexif().get(274) not in (None, 1)
                or image.mode not in {"RGB", "RGBA", "L", "LA"}
                or image.width * image.height > 64_000_000):
            raise ValueError("unsupported_raster_view_input")
        image.load()
        return image.copy()


def native_fits(width: int, height: int, *, max_edge: int, max_visual_tokens: int, patch_edge: int) -> bool:
    """Test the supplied pixel/patch capacity; the caller owns its meaning."""
    return (ceil(width / patch_edge) * patch_edge <= max_edge
            and ceil(height / patch_edge) * patch_edge <= max_edge
            and ceil(width / patch_edge) * ceil(height / patch_edge) <= max_visual_tokens)


def complete_rectangles(width: int, height: int, max_details: int, *, max_edge: int, max_visual_tokens: int,
                        patch_edge: int, overlap_fraction: float) -> tuple[tuple[int, int, int, int], ...]:
    if min(width, height, max_edge, max_visual_tokens, patch_edge) < 1 or not 0 <= overlap_fraction < 1:
        raise ValueError("invalid_raster_view_geometry")
    bounds = dict(max_edge=max_edge, max_visual_tokens=max_visual_tokens, patch_edge=patch_edge)
    if native_fits(width, height, **bounds):
        return ()
    for count in range(2, max_details + 1):
        choices = []
        for columns in range(1, count + 1):
            if count % columns:
                continue
            rows = count // columns
            if columns > width or rows > height:
                continue
            rectangles = []
            for row in range(rows):
                for column in range(columns):
                    left, right = column * width // columns, (column + 1) * width // columns
                    top, bottom = row * height // rows, (row + 1) * height // rows
                    dx, dy = ceil((right - left) * overlap_fraction / 2), ceil((bottom - top) * overlap_fraction / 2)
                    rectangles.append((max(0, left - dx), max(0, top - dy),
                                       min(width, right + dx), min(height, bottom + dy)))
            sizes = [(right - left, bottom - top) for left, top, right, bottom in rectangles]
            if all(native_fits(w, h, **bounds) for w, h in sizes):
                score = (max(ceil(w / patch_edge) * ceil(h / patch_edge) for w, h in sizes),
                         sum(w * h for w, h in sizes), columns, rows)
                choices.append((score, tuple(rectangles)))
        if choices:
            return min(choices)[1]
    raise ValueError("complete_raster_views_exceed_image_count")


def encode_region(image: Image.Image, rectangle: tuple[int, int, int, int]) -> bytes:
    with image.crop(rectangle) as region:
        buffer = BytesIO()
        region.save(buffer, format="PNG")
        return buffer.getvalue()
