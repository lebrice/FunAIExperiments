

from typing import *

from enum import Enum

class Color(Enum):
    _ = 0
    X = 1
    O = 2



class Connect4Board():
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid: List[List[Color]] = [
            [Color._ for _ in range(height)]
            for _ in range(width)
        ]
    
    def clone(self):
        clone = Connect4Board(self.width, self.height)
        for i in range(self.width):
            clone.grid[i] = list(self.grid[i])

    def make_move(self, col: int, color: Color):
        assert 0 <= col < self.width
        i = 0
        while i < self.height and self.grid[col][i] == Color._:
            i += 1
        self.grid[col][i-1] = color

    def to_txt(self):
        string = ""
        for j in range(self.height):
            for i in range(self.width):
                string += self.grid[i][j].name + " "
            string += "\n"
        return string

    def rows(self):
        for y in range(self.height):
            yield [self.grid[x][y] for x in range(self.width)]

    def columns(self):
        return self.grid


    def is_won(self) -> bool:
        """
        Detect if someone has won the board yet. (a streak of 4 of the same Color in any horizontal or diagonal direction.)
        """
        def check_horizontal(color: Color):
            for y, row in enumerate(self.rows()):
                for x_start in range(0, self.width-3):
                    x_end = x_start + 4
                    if all(cell == color for cell in row[x_start:x_end]):
                        return True
            return False

        def check_vertical(color: Color):
            for x, column in enumerate(self.columns()):
                for y_start in range(0, self.height-3):
                    y_end = y_start + 4
                    if all(cell == color for cell in column[y_start:y_end]):
                        return True
            return False
        return any(
            check_horizontal(color) or check_vertical(color)
            for color in (Color.X, Color.O)
        )







def main():
    board = Connect4Board(5,4)
    board.make_move(0, Color.O)
    board.make_move(1, Color.X)
    board.make_move(1, Color.O)
    board.make_move(2, Color.O)
    board.make_move(3, Color.X)
    board.make_move(4, Color.O)
    board.make_move(4, Color.O)
    board.make_move(4, Color.O)
    board.make_move(4, Color.X)
    print(board.to_txt())
    print(board.is_won())

if __name__ == "__main__":
    main()