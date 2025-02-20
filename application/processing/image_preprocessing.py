import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    return blurred

def edge_and_dilate(img, dilate_iterations=1):
    edges = cv2.Canny(img, 30, 120, apertureSize=3, L2gradient=True)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilate_iterations)
    return dilated

def detect_lines(edges, minLength=200, maxGap=20):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=minLength, maxLineGap=maxGap)
    if lines is None:
        raise ValueError("No lines detected in the image.")
    return lines

def extrapolate_line(x1, y1, x2, y2, img_shape):
    """Extrapolates a line to the edges of the image."""
    height, width = img_shape[:2]
    
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        y0 = int(slope * 0 + intercept)  # Left intersection
        y_max = int(slope * width + intercept)  # Right intersection
        return 0, y0, width, y_max
    else:
        return x1, 0, x2, height  # Vertical line
    
def calculate_angle(line1, line2):
    """Calculates the angle between two lines."""
    dx1, dy1 = line1[2] - line1[0], line1[3] - line1[1]
    dx2, dy2 = line2[2] - line2[0], line2[3] - line2[1]

    dot_product = dx1 * dx2 + dy1 * dy2
    mag1 = np.sqrt(dx1**2 + dy1**2)
    mag2 = np.sqrt(dx2**2 + dy2**2)

    if mag1 == 0 or mag2 == 0:
        return None

    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    return np.arccos(cos_angle) * (180.0 / np.pi)

def line_intersection(line1, line2, tolerance=5):
    """Finds the intersection of two lines if they form an angle close to 90 degrees."""
    angle = calculate_angle(line1, line2)
    # print(angle)
    
    if angle is None or ((90-tolerance) <= angle <= (90+tolerance)):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        A1, B1, C1 = float(y2 - y1), float(x1 - x2), float((y2 - y1) * x1 + (x1 - x2) * y1)
        A2, B2, C2 = float(y4 - y3), float(x3 - x4), float((y4 - y3) * x3 + (x3 - x4) * y3)

        determinant = A1 * B2 - A2 * B1

        if abs(determinant) >= 1e-10:
            try:
                x = (B2 * C1 - B1 * C2) / determinant
                y = (A1 * C2 - A2 * C1) / determinant
                return int(x), int(y)
            except OverflowError:
                return None
    return None

def detect_intersections(lines):
    """Detects intersections between detected lines."""
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = line_intersection(lines[i], lines[j])
            if intersection is not None:
                intersections.append(intersection)

    return np.array(intersections)

def find_extreme_points(intersections):
    """Finds the 4 extreme points of the chessboard using the Convex Hull."""
    hull = cv2.convexHull(intersections)
    if len(hull) < 4:
        raise ValueError("Convex Hull does not have enough points to calculate the extremes.")

    hull = np.squeeze(hull)
    rect = np.zeros((4, 2), dtype='float32')

    s = hull.sum(axis=1)
    rect[0] = hull[np.argmin(s)]  # Top-left
    rect[2] = hull[np.argmax(s)]  # Bottom-right

    diff = np.diff(hull, axis=1)
    rect[1] = hull[np.argmin(diff)]  # Top-right
    rect[3] = hull[np.argmax(diff)]  # Bottom-left

    return rect

def calculate_image_size(rect):
    """Calculates the image dimensions based on the 4 extreme points."""
    width_top = np.linalg.norm(rect[0] - rect[1])
    width_bottom = np.linalg.norm(rect[2] - rect[3])
    height_left = np.linalg.norm(rect[0] - rect[3])
    height_right = np.linalg.norm(rect[1] - rect[2])

    width = int((width_top + width_bottom) / 2)
    height = int((height_left + height_right) / 2)

    return width, height

def warp_perspective(img, corners):
    """Applies perspective transformation to the image."""
    width, height = calculate_image_size(corners)
    
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(corners, dst_points)
    
    return cv2.warpPerspective(img, M, (width, height))

def dbscan_cluster_points(intersections, eps=10, min_samples=2):
    """
    Groups points based on Euclidean distance using DBSCAN.
    intersections: np.ndarray with detected points.
    eps: maximum distance to consider points in the same cluster.
    min_samples: minimum number of points to form a cluster.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(intersections)
    labels = clustering.labels_

    # Calculate the central point of each cluster
    unique_labels = set(labels)
    clusters = []
    for label in unique_labels:
        if label != -1:  # Ignore noise
            cluster_points = intersections[labels == label]
            # centroid = cluster_points.mean(axis=0) # arithmetic mean
            # centroid = cluster_points[np.argmin(np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1))] # densest point
            centroid = np.median(cluster_points, axis=0) # median
            clusters.append(centroid)
    if (len(clusters) != 81):
        print(f"[error] Detected {len(clusters)} intersections instead of 81. Please try again using another image on better lighting conditions and less background noise.")
        exit()
    return np.array(clusters)

def group_points_by_order(points):
    """
    Organize the points by row and column order.\n
    It uses ascending x order and descending y order
    so it matches the chessboard pattern and digital image coordinates.\n
    Example:\n
    [0,0] = lowest x value and highest y value\n
    [0,8] = lowest x value and lowest y value\n
    [8,0] = highest x value and highest y value
    """
    sorted_points = points[np.lexsort((points[:, 1], points[:, 0]))] # sort points by x value
    positions = np.array_split(sorted_points, 9) # split into 9 arrays of 9 points sorted by increasing x value
    # sort each array by decreasing y value
    for i, row in enumerate(positions):
        positions[i] = sorted(row, key=lambda x: x[1], reverse=True)
    return positions