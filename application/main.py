from ifsc_chess.chessboard_operations import *
from ifsc_chess.chessboard import *
from ifsc_chess.enums import *
from detection.roboflow_detection import *
from utils.image_utils import *
import chess
import ifsc_chess.chessboard as chessboard

images = [
    "../test_images/0001.jpg", # initial
    "../test_images/0002.jpg", # e4
    "../test_images/0003.jpg", # e5
    "../test_images/0004.jpg", # Nf3
    "../test_images/0005.jpg", # d6
    "../test_images/0006.jpg", # d4
    "../test_images/0007.jpg", # Bg4
    "../test_images/0008.jpg", # dxe5
    "../test_images/0009.jpg", # Bxf3
    "../test_images/0010.jpg", # Qxf3
    "../test_images/0011.jpg", # dxe5
    "../test_images/0012.jpg", # Bc4
    "../test_images/0013.jpg", # Nf6
    "../test_images/0014.jpg", # Qb3
    "../test_images/0015.jpg", # Qe7
    "../test_images/0016.jpg", # Nc3
    "../test_images/0017.jpg", # c6
    "../test_images/0018.jpg", # Bg5
    "../test_images/0019.jpg", # b5
    "../test_images/0020.jpg", # Nxb5
    "../test_images/0021.jpg", # cxb5
    "../test_images/0022.jpg", # Bxb5+
    "../test_images/0023.jpg", # Nbd7
    # for the castle to be detected, you need the whole sequence of moves - since 0001.jpg, because of the python-chess lib's FEN notation "KQkq", which is not created if you don't build the Board object from the initial state
    "../test_images/0024.jpg", # O-O-O 
    "../test_images/0025.jpg", # Rd8
    "../test_images/0026.jpg", # Rxd7
    "../test_images/0027.jpg", # Rxd7
    "../test_images/0028.jpg", # Rd1
    "../test_images/0029.jpg", # Qe6
    "../test_images/0030.jpg", # Bxd7+
    "../test_images/0031.jpg", # Nxd7
    "../test_images/0032.jpg", # Qb8+
    "../test_images/0033.jpg", # Nxb8
    "../test_images/0034.jpg", # Rd8#
]

board_state = None
moves_list = []

def is_conversion_successful(board_matrix: chessboard.ChessboardMatrix, chessboard: chess.Board):
    fen1 = board_matrix.export_to_fen()
    fen2 = chessboard.board_fen()
    if fen1 == fen2:
        return True
    print(f"[debug] board matrix FEN: {fen1}")
    print(f"[debug] chess.Board  FEN: {fen2}")
    return False

def validate_board_matrix(fen: str):
    piece_counts = {
        'K': 0, 'Q': 0, 'R': 0, 'B': 0, 'N': 0, 'P': 0,
        'k': 0, 'q': 0, 'r': 0, 'b': 0, 'n': 0, 'p': 0
    }

    # Contar as peças no FEN
    for char in fen.split()[0]:
        if char in piece_counts:
            piece_counts[char] += 1

    # Validar o número de peças
    if piece_counts['K'] != 1:
        print(f"[error] the number of white king detected was not 1: {fen}")
        return False
    if piece_counts['k'] != 1:
        print(f"[error] the number of black king detected was not 1: {fen}")
        return False
    if piece_counts['P'] > 8:
        print(f"[error] more than 8 white pawns were detected: {fen}")
        return False
    if piece_counts['p'] > 8:
        print(f"[error] more than 8 black pawns were detected: {fen}")
        return False

    # Verificar o número total de peças
    total_pieces = sum(piece_counts.values())
    if total_pieces > 32:
        print("[error] {total_pieces} pieces detected. The board cannot have more than 32 pieces.")
        return False

    return True

def find_san_move(board_before: chess.Board, board_after: chess.Board):
    move_uci = None
    for move in board_before.legal_moves:
        board_before.push(move)
        if board_before.board_fen() == board_after.board_fen():
            move_uci = move
            break
        board_before.pop()
    if move_uci is not None:
        board_before.pop()
    return board_before.san(move)

def update_board_state(board: chess.Board, san_move: str):
    global board_state
    if board_state is None:
        board_state = board.copy()
    else:
        board_state.push_san(san_move)

def generate_board_matrix_from_image(imgPath : str, model):
    img = load_image(imgPath)
    warped = select_chessboard_area(img)
    chessboard_matrix = map_chessboard(warped)
    results = roboflow_detect_objects(model, warped)
    draw_bounding_boxes(warped, results)
    map_all_pieces(chessboard_matrix, results)
    return chessboard_matrix

if __name__ == '__main__':

    model = load_roboflow_model()

    for image in images:
        board_matrix = generate_board_matrix_from_image(image, model)
        matrix_fen = board_matrix.export_to_fen()

        if not validate_board_matrix(matrix_fen):
            print("[error] Invalid board matrix detected. Try another image. Exiting...")
            exit()

        new_board = chess.Board(fen=matrix_fen)

        if image == images[0]: # initial state
            board_state = new_board.copy()
            print(f"[info] Initial state: {board_state.fen()}")
        else:
            san_move = find_san_move(board_state, new_board)
            if san_move == None:
                print("[error] White must be first to move!")
                exit()
            print(f"[info] Detected move: {san_move}")

            update_board_state(new_board, san_move)
            if not is_conversion_successful(board_matrix, board_state):
                print("[error] Conversion from board matrix to chess.Board failed. \nThere must have been an error on the chess piece detection/classification or the sequence of images does not start with the initial board state. \nExiting...")
                exit()
            print(f"[info] New board state: {board_state.fen()}")
            moves_list.append(san_move)

    print("========================================")
    print("======== GAME FINISHED! MOVES: =========")
    for i, move in enumerate(moves_list):
        if i % 2 == 0:
            print(f"{(i // 2) + 1}. {move}", end=" ")
        else:
            print(move)
    print("\n========================================")


