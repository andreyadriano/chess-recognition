import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from application.processing.image_preprocessing import *

import numpy as np

class ChessboardCell:
    def __init__(self, bottom_left, top_right, piece=None):
        self.coordinates = (bottom_left, top_right)
        self.piece = piece

    def get_bottom_left(self):
        return self.coordinates[0]
    
    def get_top_right(self):
        return self.coordinates[1]
    
    def get_coordinates(self):
        return self.coordinates
    
    def get_piece(self):
        return self.piece
    
    def set_piece(self, piece):
        self.piece = piece
    
    def __repr__(self):
        return f"Coordinates: {self.coordinates}, Piece: {self.piece}"


class ChessboardMatrix:
    def __init__(self, points):

        if len(points) != 81:
            raise ValueError("Error: The number of points must be 81.")

        # Inicializa a matriz 8x8 com células
        self.matrix = np.zeros((8, 8), dtype=object)
        
        # Dividir os pontos em linhas
        ordered_points = group_points_by_order(points)
        
        # Preencher a matriz 8x8 com as casas do tabuleiro contendo as coordenadas de seu ponto inferior esquerdo e superior direito
        for x, array in enumerate(ordered_points):
            for y, point in enumerate(array):
                if x<=7 and y<=7:
                    bottom_left = tuple(map(int, point))
                    top_right = tuple(map(int,ordered_points[x+1][y+1]))
                    # print(f"cell [{x}][{y}] - bottom_left: {bottom_left}, top_right: {top_right}")
                    self.matrix[x, y] = ChessboardCell(bottom_left, top_right)
        
    def get_cell(self, position):
        x, y = position
        return self.matrix[x, y]
    
    def export_to_fen(self):
        """
        Exporta o estado atual do tabuleiro em formato FEN.
        """
        fen = ""
        for y in range(7, -1, -1):  # De cima para baixo (linhas 8 até 1)
            empty_count = 0
            for x in range(8):  # Da esquerda para direita (colunas a até h)
                cell = self.get_cell((x, y))
                piece = cell.get_piece()

                if piece:
                    # Adiciona o número de casas vazias antes de uma peça, se houver
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0

                    # Converte "white_king" para "K", "black_pawn" para "p", etc.
                    fen += self._piece_to_fen_symbol(piece)
                else:
                    empty_count += 1

            # Adiciona o número de casas vazias restantes no final da linha
            if empty_count > 0:
                fen += str(empty_count)

            # Adiciona a barra "/" entre as linhas, exceto na última linha
            if y > 0:
                fen += "/"

        # Adiciona os campos extras do FEN (quem joga, roques, en passant, etc.)
        if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR": # initial state
            fen += " w KQkq - 0 1"

        return fen
    
    def _piece_to_fen_symbol(self, piece):
        """
        Converte uma string como 'white_king' ou 'black_pawn' para o símbolo FEN.
        """
        piece_map = {
            "white_king": "K",
            "white_queen": "Q",
            "white_rook": "R",
            "white_bishop": "B",
            "white_knight": "N",
            "white_pawn": "P",
            "black_king": "k",
            "black_queen": "q",
            "black_rook": "r",
            "black_bishop": "b",
            "black_knight": "n",
            "black_pawn": "p",
        }
        return piece_map.get(piece, "")
