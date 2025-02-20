import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

import cv2

from application.config import SHOW_IMAGES

def show_image(img, title="output"):
    """Displays an image in a resizable window until any key is pressed."""
    if (not SHOW_IMAGES): return
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)

def load_image(image_path):
    print(f"[debug] Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"[error] Could not load image '{image_path}'.")
    return img

def save_image(img, output_path):
    cv2.imwrite(output_path, img)
    print(f"[debug] Output image saved to '{output_path}'.")

def draw_extrapolated_lines(img, lines):
    """Draws extrapolated lines on a copy of the image."""
    if (not SHOW_IMAGES): return
    img_with_lines = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    show_image(img_with_lines)

def draw_intersections(img, intersections):
    """Draws intersection points on a copy of the image."""
    if (not SHOW_IMAGES): return
    img_with_intersections = img.copy()
    for point in intersections:
        cv2.circle(img_with_intersections, tuple(point), 5, (0, 0, 255), -1)
    show_image(img_with_intersections)

def draw_filtered_points(img, points):
    """
    Draws the filtered points on the image.
    img: original image.
    points: np.ndarray with filtered points.
    title: window title.
    """
    if (not SHOW_IMAGES): return
    img_with_points = img.copy()
    for point in points:
        x, y = map(int, point)  # Ensure integer coordinates
        cv2.circle(img_with_points, (x, y), 15, (0, 0, 255), -1)  # Draw a red point
    show_image(img_with_points)

def draw_bounding_boxes(img, predictions):
    """
    Draws bounding boxes on the image.
    img: original image.
    results: list of detected objects.
    """
    if (not SHOW_IMAGES): return
    img_with_boxes = img.copy()
    color = (0,0,255)

    for bounding_box in predictions:
        x0 = bounding_box["x"] - bounding_box["width"] / 2
        x1 = bounding_box["x"] + bounding_box["width"] / 2
        y0 = bounding_box["y"] - bounding_box["height"] / 2
        y1 = bounding_box["y"] + bounding_box["height"] / 2
        cv2.rectangle(img_with_boxes, (int(x0), int(y0)), (int(x1), int(y1)), color=color, thickness=4)

        cv2.circle(img_with_boxes, (int(bounding_box["x"]), int(bounding_box["y"])), 10, color=(0,255,0), thickness=-1)

        text = f"{bounding_box['confidence']:.2f} {bounding_box['class']}"
        cv2.putText(
            img_with_boxes,
            text,
            (int(x0), int(y0)+25),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1.0,
            color = color,
            thickness=3
        )
    show_image(img_with_boxes)