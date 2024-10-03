import connect4 as c4
import json
from pathlib import Path

MOVES = 4
ALL_MOVES_COUNT = 3
DEPTH = 3
AUTOSAVE_FREQ = 5
EVALUATE = c4.evaluate_depth
BASESTATE = c4.GameState()
CUR_VERSION = "0.1.1"

path = Path(__file__).parent.absolute()

try:
    with open(path / "connect4.opening_book.json", "r") as f:
        book: dict = json.load(f)
        states = book.get("book", {})
    version = book.get("version", CUR_VERSION)
    MOVES = book.get("OMOVES", MOVES)
    ALL_MOVES_COUNT = book.get("GMOVES", ALL_MOVES_COUNT)
    DEPTH = book.get("DEPTH", DEPTH)
    print("Found opening book.")
except FileNotFoundError:
    states = {}
    version = CUR_VERSION
    print("Opening book not found. Creating new opening book.")

print(f"""Current Version: {CUR_VERSION}
Total Game States Represented: {len(states)}

Settings
Moves Ahead: {MOVES}
General Moves Ahead: {ALL_MOVES_COUNT}
Depth: {DEPTH}
Eval Function: {EVALUATE.__qualname__}""")

while True:
    inp = input("Change settings? (Y/n)").casefold()
    if inp.startswith("y"):
        r = True
        while r:
            try:
                MOVES = int(input("Moves # (optimized): "))
                r = False
            except ValueError:
                print("Invalid number.")
        r = True
        while r:
            try:
                ALL_MOVES_COUNT = int(input("Moves # (general): "))
                r = False
            except ValueError:
                print("Invalid number.")
        r = True
        while r:
            try:
                DEPTH = int(input("Eval Depth: "))
                r = False
            except ValueError:
                print("Invalid number.")
        break
    elif inp.startswith("n"):
        break


def encode_state_nh(state: c4.GameState) -> str:
    return ":".join(
        "".join("1" if disk == c4.P1DISK else "2" for disk in col)
        for col in state.board
    )


# First item is encode state function, second is if we reevaluate from that version.
old_versions = {"0.1.0": (encode_state_nh, False)}

encode_state = c4.encode_state

old_encode_state, redo = old_versions.get(version, (encode_state, False))

redo_encode = version in old_versions


def save(ver=CUR_VERSION):
    with open(path / "connect4.opening_book.json", "w") as file:
        json.dump({"version": ver, "OMOVES": MOVES, "GMOVES": ALL_MOVES_COUNT, "DEPTH": DEPTH, "book": states}, file, indent=4)


skips = 0


def evaluate_move_and_children(
    game: c4.GameState,
    moves_left: int,
    is_p1: bool = True,
    is_ai_p1: bool | None = None,
):
    if is_ai_p1 is is_p1 or is_ai_p1 is None:
        old_encoded_state = old_encode_state(game)
        encoded_state = encode_state(game)
        global skips

        if (
            redo
            or old_encoded_state not in states
            and encoded_state not in states
            or states[
                encoded_state if encoded_state in states else old_encoded_state
            ]["depth"]
            < DEPTH
        ):
            if skips:
                if skips == 1:
                    print("A calculation was skipped.")
                else:
                    print(skips, "calculations were skipped.")
                skips = 0
            print("Next:")
            print(c4.buildBoard(game, c4.NODISK))
            best_column, evaluation = EVALUATE(game, is_p1, depth=DEPTH)
            states[encoded_state] = {
                "column": best_column,
                "eval": evaluation,
                "depth": DEPTH,
            }
            if redo_encode:
                states.pop(old_encoded_state, None)
            mirror_game = c4.GameState(
                game.board[::-1], game.height, game.sequences
            )
            mirror_encode = encode_state(mirror_game)
            if (
                mirror_encode not in states
                or states[mirror_encode]["depth"] < DEPTH
                or redo_encode
            ):
                states[mirror_encode] = {
                    "column": (game.width - 1) - best_column,
                    "eval": evaluation,
                    "depth": DEPTH,
                }
                old_mirror_encode = old_encode_state(mirror_game)
                if redo_encode:
                    states.pop(old_mirror_encode, None)
            print(f"Best Column: {best_column+1}\nEvaluation:  {evaluation}")
            global calcs_without_saving
            calcs_without_saving += 1
            if not calcs_without_saving % AUTOSAVE_FREQ:
                print("Autosaving...")
                save(version)
                print("Autosave successful.")
        else:
            skips += 1
            # print("Next game state found in opening book.")
            if redo_encode and old_encoded_state in states:
                states[encoded_state] = states.pop(old_encoded_state)
            best_column = states[encoded_state]["column"]
    if moves_left:
        if is_ai_p1 is is_p1:
            evaluate_move_and_children(
                game.place_disk(best_column, c4.P1DISK if is_p1 else c4.P2DISK),
                moves_left - 1,
                not is_p1,
                is_ai_p1,
            )
        else:
            for column in range(game.width):
                if game.can_play(column):
                    evaluate_move_and_children(
                        game.place_disk(
                            column, c4.P1DISK if is_p1 else c4.P2DISK
                        ),
                        moves_left - 1,
                        not is_p1,
                        is_ai_p1,
                    )


calcs_without_saving = 0

try:
    evaluate_move_and_children(BASESTATE, MOVES, is_ai_p1=True)
    if skips:
        if skips == 1:
            print("A calculation was skipped.")
        else:
            print(skips, "calculations were skipped.")
        skips = 0
    print("Player 1 opening calculations complete.")

    evaluate_move_and_children(BASESTATE, MOVES, is_ai_p1=False)
    if skips:
        if skips == 1:
            print("A calculation was skipped.")
        else:
            print(skips, "calculations were skipped.")
        skips = 0
    print("Player 2 opening calculations complete.")

    evaluate_move_and_children(BASESTATE, ALL_MOVES_COUNT, is_ai_p1=None)
    if skips:
        if skips == 1:
            print("A calculation was skipped.")
        else:
            print(skips, "calculations were skipped.")
        skips = 0
    print("General opening calculations complete.")

    print("Saving opening book...")
    save()
except KeyboardInterrupt as e:
    print(e)
    print("Process interrupted. Saving progress...")
    save(version)

print(f"Saved {len(states)} game states successfully.")
