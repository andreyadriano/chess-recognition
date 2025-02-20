import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from application.processing.image_preprocessing import *
from application.utils.image_utils import *
from application.ifsc_chess.chessboard import *

def select_chessboard_area(img, ouput_path=None):
    """
    Selects the area of the chessboard in the image.
    img: original image.
    Returns: the warped perspective of the selected area.
    """
    show_image(img)
    pre_processed = pre_process(img)
    show_image(pre_processed)
    dilated_edges = edge_and_dilate(pre_processed, 2)
    show_image(dilated_edges)
    lines = detect_lines(dilated_edges)
    extrapolated_lines = [extrapolate_line(*line[0], img.shape) for line in lines]
    intersections = detect_intersections(extrapolated_lines)
    draw_extrapolated_lines(img, extrapolated_lines)
    draw_intersections(img, intersections)
    corners = find_extreme_points(intersections)
    warped = warp_perspective(img, corners)
    show_image(warped)

    if ouput_path is not None:
        save_image(warped, ouput_path)

    return warped


def map_chessboard(img):
    """
    Maps the chessboard grid to a 8x8 matrix
    img: selected area of the chessboard.
    Returns: the 8x8 matrix.
    """
    show_image(img)

    pre_processed = pre_process(img)
    show_image(pre_processed)

    dilated_edges = edge_and_dilate(pre_processed)
    show_image(dilated_edges)

    lines = detect_lines(dilated_edges)
    extrapolated_lines = [extrapolate_line(*line[0], img.shape) for line in lines]

    intersections = detect_intersections(extrapolated_lines)
    filtered_intersections = dbscan_cluster_points(intersections, eps=img.shape[0]/12, min_samples=5)

    draw_extrapolated_lines(img, extrapolated_lines)
    draw_intersections(img, intersections)
    draw_filtered_points(img, filtered_intersections)

    return ChessboardMatrix(filtered_intersections)

def map_single_piece(matrix: ChessboardMatrix, piece):
    """
    Map a piece based on its bounding box center
    """
    center = piece["x"], piece["y"]
    # print(f"Piece {piece["class"]} center = {center}")
    for column in range(8):
        top_right = matrix.get_cell((column, 0)).get_top_right()
        if center[0] <= top_right[0]:
            for row in range(8):
                top_right = matrix.get_cell((column, row)).get_top_right()
                if center[1] >= top_right[1]:
                    # print(f"Piece {piece["class"]} is in cell [{column},{row}]")
                    matrix.get_cell((column, row)).set_piece(piece["class"])
                    return True

def map_all_pieces(matrix: ChessboardMatrix, pieces):
    """
    Maps all pieces detected in the image to the chessboard matrix.
    results: Roboflow YOLO model output
    Returns: the chessboard matrix with the pieces mapped.
    """
    for piece in pieces:
        if not map_single_piece(matrix, piece):
            print(f"[error] Piece {piece['class']} couldn't be mapped to the chessboard.")
            return False
    return True