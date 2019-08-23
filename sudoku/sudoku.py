"""
Fabrice Normandin - fabrice.normandin@gmail.com

Algorithm to solve sudokus.


For fun, while on the plane from Istanbul to Berlin
(finished while on the way to Montreal from London)
"""

from time import time
from typing import *
from queue import PriorityQueue

import multiprocessing as mp

SudokuGrid = List[List[Optional[int]]]


class Board():
    def __init__(self, grid: SudokuGrid):
        width = len(grid)
        height = len(grid[0])

        self.width = width
        self.height = height

        self.grid: SudokuGrid = [
            [None for _ in range(width)]
            for _ in range(height)
        ]

        self.empty_cells: Set[Tuple[int, int]] = set(
            (i, j) for i in range(height) for j in range(width))
        self.possibilities: List[List[Set[int]]] = [
            [
                set(range(1, 10))
                for _ in range(width)
            ] for _ in range(height)
        ]

        self._setup_state(grid)

    def _setup_state(self, grid):
        for row, row_values in enumerate(grid):
            for col, value in enumerate(row_values):
                # print(f"row: {row}, col: {col}")
                if value is None:
                    # print(f"Empty cell at {row}, {col}")
                    self.empty_cells.add((row, col))
                else:
                    self.update(row, col, value)

    @classmethod
    def empty(cls, width=9, height=9) -> "Board":
        """
        Creates a new, empty board.
        """
        return cls(grid=[
            [None for _ in range(width)]
            for _ in range(height)
        ])

    def update(self, row: int, col: int, new_value: int) -> None:
        """
        We fill an empty spot on the board, and update the corresponding state buffers for the columns, rows, and squares.
        """
        # print(f"Writing the value {new_value} at position ({row}, {col})")
        if new_value is not None:
            assert (row, col) in self.empty_cells
            assert self.grid[row][col] is None
            assert new_value in self.possibilities[row][col]

            self.grid[row][col] = new_value
            self.empty_cells.discard((row, col))

            # update the possible values in the squares of the same column
            for i in range(self.height):
                self.possibilities[i][col].discard(new_value)

            # update the possible values in the squares of the same row
            for j in range(self.width):
                self.possibilities[row][j].discard(new_value)

            # update the possible values in the same square
            for i in range((row // 3 * 3), (row // 3 + 1) * 3):
                for j in range((col // 3 * 3), ((col // 3 + 1) * 3)):
                    if self.grid[i][j] is None:
                        self.possibilities[i][j].discard(new_value)

        else:
            raise NotImplementedError("Erasing a value isnt implemented yet.")

    def obvious_move(self) -> Optional[Tuple[int, int, int]]:
        for (row, col) in self.empty_cells:
            if len(self.possibilities[row][col]) == 1:
                value = next(iter(self.possibilities[row][col]))
                return row, col, value
        return None

    def solve_obvious_moves(self) -> int:
        "returns the number of obvious moves that were performed."
        assert self.valid
        count = 0
        move = self.obvious_move()
        while move is not None:
            row, col, new_value = move
            # print(f"Placing a {new_value}, at position ({row}, {col})")
            self.update(*move)
            move = self.obvious_move()
            count += 1
        return count

    def copy(self) -> "Board":
        return Board(self.grid)

    @classmethod
    def read_csv(cls, file_path: str) -> "Board":
        with open(file_path) as file:
            return cls.from_string(file.read())

    def write_csv(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            file.write(str(self))

    def __str__(self) -> str:
        string = ""
        for row in self.grid:
            string += ",".join(str(c)
                               if c is not None else "-" for c in row) + "\n"
        return string

    @classmethod
    def from_string(cls, string: str) -> "Board":
        grid = []
        for line in string.splitlines():
            parts = line.split(",")
            grid.append([None if p == "-" else int(p) for p in parts])
        return Board(grid)

    def valid(self) -> bool:
        """
        Checks that the Board is in a valid state (no two same numbers in any row, column, or square.)
        """
        rows = [
            set(range(1, 10)) for _ in range(self.height)
        ]
        cols = [
            set(range(1, 10)) for _ in range(self.width)
        ]
        squares = [
            [
                set(range(1, 10)) for _ in range(self.width // 3)
            ] for _ in range(self.height // 3)
        ]
        try:
            for i, row in enumerate(self.grid):
                for j, value in enumerate(row):
                    if value is not None:
                        rows[i].remove(value)
                        cols[j].remove(value)
                        squares[i//3][j//3].remove(value)
        except KeyError:
            return False
        else:
            return True

    def filled(self) -> bool:
        return all(value is not None for row in self.grid for value in row)

    def solved(self) -> bool:
        return self.filled() and self.valid()

    def __lt__(self, other: "Board"):
        """
        Used as part of the PriorityQueue mechanism, when two boards have the same priority.
        We don't really care about how two boards with the same number of empty cells are compared, hence we just always return True here. 
        """
        return True

    def num_possibilities(self, coords: Tuple[int, int]) -> int:
        """Helper function. Returns the number of possible options for a given empty square on the board.
        """
        i, j = coords
        return len(self.possibilities[i][j])

    def variants(self) -> Iterable["Board"]:
        """
        For a given board, this function returns a generator that yields modified copies of the board where one possible guess was taken.
        Starts by guessing in squares with a low number of possibilities first.
        """
        # the number of empty cells could also be an indicator of recursion, potentially.
        # num_empty = len(self.empty_cells)

        # choose the empty squares with the fewest possibilities first
        for row, col in sorted(self.empty_cells, key=self.num_possibilities):
            for possible_value in self.possibilities[row][col]:
                # print(f"{num_empty}: Attempting the guess of putting a {possible_value} at position {empty_square}")
                new_board = self.copy()
                new_board.update(row, col, possible_value)  # set the value
                yield new_board
                # print(f"{num_empty}: This guess of putting a {possible_value} at position {empty_square} didn't work.")


def a_star_search(queue) -> Board:
    """
    The actual 'search' portion of the algorithm.
    I think its some form of A* search, when the queue used is a priority queue.
    """
    while not queue.empty():
        _, most_filled_board = queue.get()
        for board_variant in most_filled_board.variants():
            board_variant.solve_obvious_moves()
            if board_variant.solved():
                print("YAY! Board is solved!")
                return board_variant
            # enqueue the unsolved board, using the number of empty cells as the priority index (lower is better)
            queue.put((len(board_variant.empty_cells), board_variant))

    raise Exception(
        "Unable to solve the given board. Are you sure it is valid and possible to solve?")


def solve_sudoku(board: Board, use_multiprocessing=False) -> Board:
    """
    TODO: the main disadvantage of using multiprocessing atm is that there doesn't seem to be a priority queue mechanism.
    Therefore, we are not dequeuing the boards in order of priority. This makes it considerably slower.
    """
    board.solve_obvious_moves()
    if board.solved():
        print("This simple sudoku was easily solved without the need for any guesses.")
        return board

    print("Obvious moves exhausted. Empty squares left:", len(board.empty_cells))

    # We need to take potentially more than one guesses in order to reach a solution.

    if use_multiprocessing:
        pool = mp.Pool()
        with mp.Pool() as pool:
            m = mp.Manager()
            q = m.Queue()
            # q = MultiProcessingPriorityQueue()
            q.put((len(board.empty_cells), board))
            results = pool.apply_async(a_star_search, args=(q,))
            board = results.get()
    else:
        # Here, I'm doing a sort of BFS/A* search, using the number of remaining empty cells as an indicator of priority for each board.
        # Therefore, the less empty squares there are, the "closer" we are to a good solution.

        q = PriorityQueue()
        # q = MultiProcessingPriorityQueue()

        q.put((len(board.empty_cells), board))
        board = a_star_search(q)

    return board


class MultiProcessingPriorityQueue():
    """
    My attempt at a multiprocessing-compatible priority queue.
    """

    def __init__(self):
        self._queues: Dict[int, List[Any]] = dict()
        self._len = 0

    def put(self, item: Tuple[int, Any]) -> None:
        priority, obj = item
        if priority not in self._queues.keys():
            self._queues[priority] = list()
        self._queues[priority].append(item)
        self._len += 1

    def get(self) -> Any:
        lowest_priority = sorted(self._queues.keys())[0]
        item = self._queues[lowest_priority].pop()
        if len(self._queues[lowest_priority]) == 0:
            del self._queues[lowest_priority]
        self._len -= 1
        return item

    def empty(self) -> bool:
        return self._len == 0


def main(filename: str):
    start = time()
    file_path = f"{filename}.csv"
    board = Board.read_csv(file_path)
    solved = solve_sudoku(board)
    print(f"\nSolved in {time() - start} seconds!")
    print(solved)
    solved.write_csv(f"{filename}_solved.csv")


if __name__ == "__main__":
    main("sudoku_very_hard")
