"""Smart image cropping with face-aware positioning."""

from __future__ import annotations

from pathlib import Path


_CASCADE_NAMES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
    "haarcascade_upperbody.xml",
]

_HEAD_MARGIN_RATIO = 0.7


class FaceInfo:
    """Detected face region (all values normalised 0-1)."""

    __slots__ = ("cx", "cy", "top", "bottom")

    def __init__(self, cx: float, cy: float, top: float, bottom: float) -> None:
        self.cx = cx
        self.cy = cy
        self.top = top
        self.bottom = bottom


def detect_face_center(
    image_path: str | None,
    *,
    cv_img: object | None = None,
) -> tuple[float, float] | None:
    """Return normalised (cx, cy) of the dominant face, or None.

    Tries multiple cascades (frontal, profile, upper body) for better
    detection of angled/side faces. Pass *cv_img* (a numpy BGR array)
    to skip file I/O.
    """
    info = detect_face_info(image_path, cv_img=cv_img)
    if info is None:
        return None
    return (info.cx, info.cy)


def detect_face_info(
    image_path: str | None,
    *,
    cv_img: object | None = None,
) -> FaceInfo | None:
    """Return detailed face region info, or None if no face detected."""
    try:
        import cv2

        if cv_img is not None:
            img = cv_img
        else:
            img = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        haar_dir = Path(cv2.data.haarcascades)  # type: ignore[attr-defined]

        best_face: tuple[int, int, int, int] | None = None
        best_area = 0

        for name in _CASCADE_NAMES:
            cascade_path = haar_dir / name
            if not cascade_path.exists():
                continue
            cascade = cv2.CascadeClassifier(str(cascade_path))
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
            )
            if len(faces) == 0:
                continue
            for x, y, w, h in faces:
                area = w * h
                if area > best_area:
                    best_area = area
                    best_face = (x, y, w, h)

        if best_face is None:
            return None
        x, y, w, h = best_face
        img_h, img_w = img.shape[:2]
        head_margin = int(h * _HEAD_MARGIN_RATIO)
        top_px = max(0, y - head_margin)
        return FaceInfo(
            cx=(x + w / 2) / img_w,
            cy=(y + h / 2) / img_h,
            top=top_px / img_h,
            bottom=(y + h) / img_h,
        )
    except Exception:
        return None


def smart_crop_box(
    iw: int,
    ih: int,
    box_w: int,
    box_h: int,
    face_cy: float | None = None,
    *,
    face_info: FaceInfo | None = None,
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) crop box that fills *box_w x box_h* ratio.

    * Horizontal: always center-crop.
    * Vertical: uses face-aware anchoring with head protection to avoid
      cutting off heads. Falls back to center-crop if no face detected.
    """
    box_ratio = box_w / box_h
    img_ratio = iw / ih

    if img_ratio > box_ratio:
        new_w = int(ih * box_ratio)
        x0 = (iw - new_w) // 2
        return (x0, 0, x0 + new_w, ih)

    new_h = int(iw / box_ratio)

    if face_info is not None:
        face_top_px = int(ih * face_info.top)
        face_bottom_px = int(ih * face_info.bottom)

        y0 = face_top_px
        y0 = max(0, min(y0, ih - new_h))

        if y0 + new_h < face_bottom_px:
            y0 = max(0, face_bottom_px - new_h)
            y0 = min(y0, ih - new_h)

    elif face_cy is not None:
        anchor_y = int(ih * face_cy)
        y0 = anchor_y - new_h * 2 // 3
        y0 = max(0, min(y0, ih - new_h))
    else:
        y0 = (ih - new_h) // 2
        y0 = max(0, min(y0, ih - new_h))

    return (0, y0, iw, y0 + new_h)
