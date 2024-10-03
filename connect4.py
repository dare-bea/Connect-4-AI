from collections.abc import Iterable
from typing import Self, Any, Literal
from itertools import product
import operator
from functools import cache, partial
import json
from pathlib import Path


class GameState:
    def __init__(
        self,
        board: Iterable[Iterable[Any]] | int = 7,
        /,
        height: int = 6,
        winning_sequences: Iterable[Iterable[tuple[int, int]]]
        | int
        | None = None,
    ) -> None:
        processed_board: list[list[Any]]
        if isinstance(board, int):
            processed_board = [[] for _ in range(board)]
        else:
            processed_board = [list(i) for i in board]
        self.board: list[list[Any]] = processed_board
        self.height: int = height
        self.width = len(processed_board)
        if winning_sequences is None:
            winning_sequences = 4
        sequences: (
            list[zip[tuple[int, int]]] | Iterable[Iterable[tuple[int, int]]]
        )
        if isinstance(winning_sequences, int):
            length: int = winning_sequences
            sequences = (
                tuple(
                    tuple(zip(range(wx, wx + length), [wy] * length))
                    for wx, wy in product(
                        range(self.width - length + 1), range(height)
                    )
                )
                + tuple(
                    tuple(zip([wx] * length, range(wy, wy + length)))
                    for wx, wy in product(
                        range(self.width), range(height - length + 1)
                    )
                )
                + tuple(
                    tuple(zip(range(wx, wx + length), range(wy, wy + length)))
                    for wx, wy in product(
                        range(self.width - length + 1),
                        range(height - length + 1),
                    )
                )
                + tuple(
                    tuple(
                        zip(
                            range(wx + length - 1, wx - 1, -1),
                            range(wy, wy + length),
                        )
                    )
                    for wx, wy in product(
                        range(self.width - length + 1),
                        range(height - length + 1),
                    )
                )
            )
        else:
            sequences = tuple(tuple(coords) for coords in winning_sequences)
        self.sequences = sequences

    def can_play(self, col: int) -> bool:
        return 0 <= col < len(self.board) and len(self.board[col]) < self.height

    def place_disk(self, col_index: int, disk: Any) -> Self:
        if not self.can_play(col_index):
            raise ValueError(f"Can't play in column {col_index}.")
        new_board = []
        for column in self.board:
            new_board.append(column.copy())
        new_board[col_index].append(disk)
        return type(self)(new_board, self.height, self.sequences)

    def winner(self) -> Literal[1, -1, 0] | None:
        """If a winning sequence is found, returns 1 or -1 depending on the winner.
        If the board is completely filled up without a winner (a draw), returns 0.
        Otherwise, returns None, indicating an active game."""
        for coords in self.sequences:
            sequence = [
                self.board[x][y] if y < len(self.board[x]) else None
                for x, y in coords
            ]
            if all(disk == P1DISK for disk in sequence):
                return 1
            if all(disk == P2DISK for disk in sequence):
                return -1
        if all(self.height == len(col) for col in self.board):
            return 0
        return None

    def __repr__(self) -> str:
        return f"GameState({self.board}, height={self.height}, sequences={self.sequences})"

    def count_disks(self) -> int:
        return sum(len(column) for column in self.board)

    def __str__(self) -> str:
        return f"<GameState object with {self.count_disks()} disks placed>"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return (
            self.board == other.board
            and self.height == other.height
            and self.width == other.width
            and self.sequences == other.sequences
        )
    
    def __hash__(self) -> int:
        return hash((
            tuple(tuple(col) for col in self.board),
            self.height,
            self.width,
            self.sequences
        ))



def buildBoard(game_state: GameState, default: str = "") -> str:
    max_length = max(
        len(str(game_state.width)),
        len(default),
        *(len(str(disk)) for col in game_state.board for disk in col),
    )
    rows: list[dict[int, list[Any]]] = [{} for _ in range(game_state.height)]
    for x, col in enumerate(game_state.board):
        for y, disk in enumerate(col):
            rows[y][x] = disk
    rows.reverse()

    matrix: list[list[Any]] = [
        *(
            [
                str(row.get(x, default)).center(max_length)
                for x in range(game_state.width)
            ]
            for row in rows
        ),
        [str(y + 1).center(max_length) for y in range(game_state.width)],
    ]
    return "\n".join(" ".join(row) for row in matrix)


def encode_state(state: GameState) -> str:
    return f"#{str(state.height)}:" + ":".join(
        "".join("1" if disk == P1DISK else "2" for disk in col)
        for col in state.board
    )

@cache
def score(game: GameState, is_p1=None) -> int:
    @cache
    def evaluate_sequence(sequence: tuple) -> int:
        count_p1 = sequence.count(P1DISK)
        count_p2 = sequence.count(P2DISK)
        
        if count_p1 > 0 and count_p2 == 0:
            return [0, 1, 4, 10 + 10 * (is_p1 is False), 200][count_p1]
        elif count_p2 > 0 and count_p1 == 0:
            return -[0, 1, 4, 10 + 10 * (is_p1 is True), 200][count_p2]
        return 0

    return sum(
        evaluate_sequence(
            tuple(game.board[x][y] if y < len(game.board[x]) else None for x, y in coords)
        )
        for coords in game.sequences
    )


def evaluate_depth(
    game: GameState, is_p1: bool, depth: int = 1
) -> tuple[int, int | float]:
    comp_func = operator.lt if is_p1 else operator.gt
    player_disk = P1DISK if is_p1 else P2DISK
    opponent_disk = P2DISK if is_p1 else P1DISK
    width = game.width

    best_self_score: int | float = float("-inf") if is_p1 else float("inf")
    best_column: int = 0

    for self_col in range(width):
        if not game.can_play(self_col):
            continue

        self_mv = game.place_disk(self_col, player_disk)
        if self_mv.winner() is not None:
            best_self_score = score(self_mv, is_p1)
            best_column = self_col
            continue

        best_oppo_score: int | float = float("inf") if is_p1 else float("-inf")
        for oppo_col in range(width):
            if not self_mv.can_play(oppo_col):
                continue

            oppo_mv = self_mv.place_disk(oppo_col, opponent_disk)
            if oppo_mv.winner() is not None:
                best_oppo_score = score(oppo_mv, is_p1)
                continue

            if depth > 1:
                this_score = evaluate_depth(oppo_mv, is_p1, depth=depth - 1)[1]
            else:
                this_score = score(oppo_mv, is_p1)

            if best_oppo_score is None or comp_func(
                this_score, best_oppo_score
            ):
                best_oppo_score = this_score
            if best_self_score is not None and comp_func(
                best_oppo_score, best_self_score
            ):
                break

        if best_self_score is None or comp_func(
            best_self_score, best_oppo_score
        ):
            best_self_score, best_column = best_oppo_score, self_col

    return best_column, best_self_score


P1DISK: str = "(R)"
P2DISK: str = "(Y)"
NODISK: str = " - "

path = Path(__file__).parent.absolute() / "connect4.opening_book.json"


def evaluate_book(
    game: GameState, is_p1: bool, book: dict, depth: int = 1
) -> tuple[int, int | float]:
    page: dict | None = book.get(encode_state(game), None)
    if page is None or page["depth"] < depth:
        result = evaluate_depth(game, is_p1, depth)
        new_pages[encode_state(game)] = {
            "column": result[0],
            "eval": result[1],
            "depth": depth,
        }
        return result
    else:
        return page["column"], page["eval"]


if __name__ == "__main__":
    try:
        with open(path, "r") as f:
            book_full = json.load(f)
            book = book_full["book"]
            print("Found opening book.")
    except FileNotFoundError:
        book_full = {"version": None, "book": {}}
        book = {}

    new_pages: dict[str, dict] = {}

    print("Choose a mode:\n - Player-1\n - Player-2\n - Bot-Only")
    MODE: Literal["Player-1", "Player-2"] | str = input(">>> ")
    SHOWSTATS: bool = True
    SHOWPLAYEREVAL: bool = False
    MIN_DEPTH = 3
    DEPTH: int = max(book_full.get("DEPTH", MIN_DEPTH), MIN_DEPTH)

    evaluate = partial(evaluate_book, book=book, depth=DEPTH)

    # game = GameState([[],[],[],[P2DISK],[],[],[]])
    game = GameState()
    history = [game]

    print(buildBoard(game, NODISK))
    if SHOWSTATS:
        print("Score:", score(game))

    while game.winner() is None:
        if MODE != "Player-1":
            col, evaluation = evaluate(game, True)
            game = game.place_disk(col, P1DISK)
        else:
            r = True
            while r:
                try:
                    inp = input("Choose a column.")
                    if inp == "undo":
                        game, history = history[-3], history[:-2]
                        print(buildBoard(game, NODISK))
                        if SHOWSTATS:
                            print("Score:", score(game))
                            if SHOWPLAYEREVAL:
                                print("Eval: ", evaluation)
                    else:
                        col = int(inp) - 1
                        if col not in range(game.width):
                            print("Column out of range.")
                        elif not game.can_play(col):
                            print("Column is full.")
                        else:
                            r = False
                except ValueError:
                    print("Invalid column.")
            game = game.place_disk(col, P1DISK)
            if SHOWPLAYEREVAL:
                evaluation = evaluate(game, True)[1]
        history.append(game)

        print(buildBoard(game, NODISK))
        if SHOWSTATS:
            print("Score:", score(game))
            if MODE == "Player-1" and SHOWPLAYEREVAL:
                print("Eval: ", evaluation)

        if game.winner() is not None:
            break

        if MODE != "Player-2":
            col, evaluation = evaluate(game, False)
            game = game.place_disk(col, P2DISK)
        else:
            r = True
            while r:
                try:
                    inp = input("Choose a column.")
                    if inp == "undo":
                        game, history = history[-3], history[:-2]
                        if SHOWSTATS:
                            print("Score:", score(game))
                            if SHOWPLAYEREVAL:
                                print("Eval: ", evaluation)
                    else:
                        col = int(inp) - 1
                        if col not in range(game.width):
                            print("Column out of range.")
                        elif not game.can_play(col):
                            print("Column is full.")
                        else:
                            r = False
                except ValueError:
                    print("Invalid column.")
            game = game.place_disk(col, P2DISK)
            if SHOWPLAYEREVAL:
                evaluation = evaluate(game, False)[1]
        history.append(game)

        print(buildBoard(game, NODISK))
        if SHOWSTATS:
            print("Score:", score(game))
            if MODE == "Player-2" and SHOWPLAYEREVAL:
                print("Eval: ", evaluation)

    print("Game over!")

    if new_pages:
        while True:
            inp = input(
                f"{len(new_pages)} new calculation(s) made. Export calculations? (Y/n)"
            ).casefold()
            if inp.startswith("y"):
                book.update(new_pages)
                book_full["book"] = book
                with open(path, "w") as f:
                    json.dump(book_full, f, indent=4)
                print("Save successful!")
                break
            elif inp.startswith("n"):
                break
