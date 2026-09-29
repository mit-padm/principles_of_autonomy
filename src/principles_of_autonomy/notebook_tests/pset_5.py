import unittest
import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
# from nose.tools import assert_equal

from principles_of_autonomy.grader import get_locals
import numpy as np

import os
import time
import tempfile
from pyperplan.pddl.parser import Parser
from pyperplan import grounding, planner

# Function for tests
def test_ok():
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Test passed!!</strong>
        </div>""", raw=True)
    except:
        print("test ok!!")


# ---------------------------------------------------------------------------
# XA: modeling an imperfect opponent.
# The student plays o (second) against a hidden x opponent. Boards are tuples
# of 9 symbols ('x', 'o', ' '), row by row.
# ---------------------------------------------------------------------------
import random
from collections import Counter
from functools import lru_cache

_WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
_XA_NUM_GAMES = 100
_XA_SEED = 0
_XA_MIN_WINS = 21  # "more than 20/100"; the intended model averages ~35 wins

# Hidden opponent: always opens in corner 0, answers o's center with edge 5 or
# 7, then plays perfectly, except in the two (mirror-image) positions below,
# where with probability _XA_BLUNDER_PROB it makes a threat that forces o to
# block into a fork.
_XA_BLUNDER_PROB = 0.35
_XA_TRAPS = {
    ('x', ' ', ' ', ' ', 'o', 'x', ' ', ' ', 'o'): 3,
    ('x', ' ', ' ', ' ', 'o', ' ', ' ', 'x', 'o'): 1,
}


def _winner(b):
    for i, j, k in _WIN_LINES:
        if b[i] != ' ' and b[i] == b[j] == b[k]:
            return b[i]
    return None


def _empty(b):
    return [i for i in range(9) if b[i] == ' ']


def _is_terminal(b):
    return _winner(b) is not None or ' ' not in b


def _final_score(b):
    w = _winner(b)
    return 1 if w == 'x' else -1 if w == 'o' else 0


def _x_to_move(b):
    return b.count('x') == b.count('o')


def _play(b, i):
    return b[:i] + ('x' if _x_to_move(b) else 'o',) + b[i + 1:]


@lru_cache(maxsize=None)
def _value(b):
    if _is_terminal(b):
        return _final_score(b)
    vals = [_value(_play(b, i)) for i in _empty(b)]
    return max(vals) if _x_to_move(b) else min(vals)


def _opponent_move(b, rng):
    num_moves = 9 - len(_empty(b))
    if num_moves == 0:
        return 0
    if num_moves == 2 and b[0] == 'x' and b[4] == 'o':
        return rng.choice([5, 7])
    if b in _XA_TRAPS and rng.random() < _XA_BLUNDER_PROB:
        return _XA_TRAPS[b]
    moves = _empty(b)
    best = max(_value(_play(b, i)) for i in moves)
    return rng.choice([i for i in moves if _value(_play(b, i)) == best])


def _make_student_policy(opponent_model, game_state, tic_tac_toe_board):
    """o's policy: expectimax where x nodes are weighted by the student's model
    and o minimizes. Ties go to the lowest-numbered cell (the first successor)."""

    @lru_cache(maxsize=None)
    def model(b):
        state = game_state(tic_tac_toe_board(list(b)))
        succs = state.successors()
        probs = list(opponent_model(state))
        board_str = list(b)
        assert len(probs) == len(succs), \
            f"opponent_model returned {len(probs)} probabilities for {board_str}, which has {len(succs)} successors."
        assert all(p >= 0 for p in probs), f"opponent_model returned a negative probability for {board_str}: {probs}"
        assert abs(sum(probs) - 1) < 1e-6, f"opponent_model probabilities for {board_str} sum to {sum(probs)}, not 1."
        return [(tuple(s.board.moves), p) for s, p in zip(succs, probs)]

    @lru_cache(maxsize=None)
    def value(b):
        if _is_terminal(b):
            return _final_score(b)
        if _x_to_move(b):
            return sum(p * value(s) for s, p in model(b))
        return min(value(_play(b, i)) for i in _empty(b))

    def policy(b):
        moves = _empty(b)
        vals = [value(_play(b, i)) for i in moves]
        best = min(vals)
        return next(i for i, v in zip(moves, vals) if v <= best + 1e-9)

    return policy


def _play_games(policy, num_games, seed):
    """Returns a list of (moves, result) where moves are cell indices, x first."""
    rng = random.Random(seed)
    games = []
    for _ in range(num_games):
        b, moves = (' ',) * 9, []
        while not _is_terminal(b):
            i = _opponent_move(b, rng) if _x_to_move(b) else policy(b)
            moves.append(i)
            b = _play(b, i)
        result = {1: 'loss', 0: 'tie', -1: 'win'}[_final_score(b)]
        games.append((tuple(moves), result))
    return games


def _format_moves(moves):
    return ' '.join(('X' if k % 2 == 0 else 'O') + str(i) for k, i in enumerate(moves))


def _format_games(games):
    """Every game in the order played, one per line, for pasting into an LLM."""
    num_wins = sum(result == 'win' for _, result in games)
    lines = [
        f"You (O) won {num_wins} of {len(games)} games (need more than {_XA_MIN_WINS - 1}).",
        "Cells are numbered row by row: 0 1 2 / 3 4 5 / 6 7 8. X (opponent) moves first; you are O.",
        "game  result  moves",
    ]
    for n, (moves, result) in enumerate(games, 1):
        lines.append(f"{n:4d}  {result:6s}  {_format_moves(moves)}")
    return '\n'.join(lines)


class TestPSet5(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(50)
    def test_1_minimax(self):
        fmin, fmax, game_state, tic_tac_toe_board = get_locals(
            self.notebook_locals, ["minimize_score", "maximize_score", "game_state", "tic_tac_toe_board"])
        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ', 'o']))
        result = fmax(test_state)
        assert isinstance(result, tuple) and len(
            result) == 3, "Result should be a 3-tuple"
        assert result[0] == 0, "From ['x','o','x','x','x','o','o',' ','o'] max score should be 0."
        assert result[1] == 2, "From ['x','o','x','x','x','o','o',' ','o'] number of explored states should be 2."
        assert len(
            result[2]) == 2, "From ['x','o','x','x','x','o','o',' ','o'] optimal play should have 2 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', ' ', 'o', 'o', ' ', 'o']))
        result = fmax(test_state)
        assert result[0] == 0, "From ['x','o','x','x',' ','o','o',' ','o'] max score should be 0."
        assert result[1] == 5, "From ['x','o','x','x',' ','o','o',' ','o'] number of explored states should be 5."
        assert len(
            result[2]) == 3, "From ['x','o','x','x',' ','o','o',' ','o'] optimal play should have 3 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', ' ', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', ' ', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."
        assert result[2][2].board.moves == ['x', 'o', 'x', 'x', 'o', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', ' ', 'x', ' ', ' ', 'o', ' ', ' ']))
        result = fmax(test_state)
        assert result[0] == 1, "From ['x','o',' ','x',' ',' ','o',' ',' '] max score should be 1."
        assert result[1] == 246, "From ['x','o',' ','x',' ',' ','o',' ',' '] number of explored states should be 246."
        assert len(
            result[2]) == 4, "From ['x','o',' ','x',' ',' ','o',' ',' '] optimal play should have 4 states."
        assert result[2][0].board.moves == ['x', 'o', ' ', 'x', ' ', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', ' ', 'x', 'x', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."
        assert result[2][2].board.moves == ['x', 'o', 'o', 'x', 'x', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."
        assert result[2][3].board.moves == ['x', 'o', 'o', 'x', 'x', 'x', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ', 'o']))
        test_state.player = 2
        result = fmin(test_state)
        assert isinstance(result, tuple) and len(
            result) == 3, "Result should be a 3-tuple"
        assert result[0] == - \
            1, "From ['x','o','x','x','x','o','o',' ','o'] min score should be -1."
        assert result[1] == 2, "From ['x','o','x','x','x','o','o',' ','o'] number of explored states should be 2."
        assert len(
            result[2]) == 2, "From ['x','o','x','x','x','o','o',' ','o'] optimal play should have 2 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', 'o',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['o', 'x', 'o', 'o', ' ', 'x', 'x', ' ', 'x']))
        test_state.player = 2
        result = fmin(test_state)
        assert result[0] == 0, "From ['o','x','o','o',' ','x','x',' ','x'] min score should be 0."
        assert result[1] == 5, "From ['o','x','o','o',' ','x','x',' ','x'] number of explored states should be 5."
        assert len(
            result[2]) == 3, "From ['o','x','o','o',' ','x','x',' ','x'] optimal play should have 3 states."
        assert result[2][0].board.moves == ['o', 'x', 'o', 'o', ' ', 'x', 'x', ' ',
                                            'x'], "From ['o','x','o','o',' ','x','x',' ','x'] incorrect optimal play."
        assert result[2][1].board.moves == ['o', 'x', 'o', 'o', ' ', 'x', 'x', 'o',
                                            'x'], "From ['o','x','o','o',' ','x','x',' ','x'] incorrect optimal play."
        assert result[2][2].board.moves == ['o', 'x', 'o', 'o', 'x', 'x', 'x', 'o',
                                            'x'], "From ['o','x','o','o',' ','x','x',' ','x'] incorrect optimal play."
        test_ok()

    @weight(25)
    def test_2_alpha_beta(self):
        fmin, fmax, game_state, tic_tac_toe_board = get_locals(
            self.notebook_locals, ["minimize_score_alpha_beta", "maximize_score_alpha_beta", "game_state", "tic_tac_toe_board"])

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ', 'o']))
        result = fmax(test_state, -2, 2)
        assert isinstance(result, tuple) and len(
            result) == 3, "Result should be a 3-tuple"
        assert result[0] == 0, "From ['x','o','x','x','x','o','o',' ','o'] max score should be 0."
        assert result[1] == 2, "From ['x','o','x','x','x','o','o',' ','o'] number of explored states should be 2."
        assert len(
            result[2]) == 2, "From ['x','o','x','x','x','o','o',' ','o'] optimal play should have 2 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', ' ', 'o', 'o', ' ', 'o']))
        result = fmax(test_state, -2, 2)
        assert result[0] == 0, "From ['x','o','x','x',' ','o','o',' ','o'] max score should be 0."
        assert result[1] == 5, "From ['x','o','x','x',' ','o','o',' ','o'] number of explored states should be 5."
        assert len(
            result[2]) == 3, "From ['x','o','x','x',' ','o','o',' ','o'] optimal play should have 3 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', ' ', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', ' ', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."
        assert result[2][2].board.moves == ['x', 'o', 'x', 'x', 'o', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', ' ', 'x', ' ', ' ', 'o', ' ', ' ']))
        result = fmax(test_state, -2, 2)
        assert result[0] == 1, "From ['x','o',' ','x',' ',' ','o',' ',' '] max score should be 1."
        assert result[1] == 89, "From ['x','o',' ','x',' ',' ','o',' ',' '] number of explored states should be 89."
        assert len(
            result[2]) == 4, "From ['x','o',' ','x',' ',' ','o',' ',' '] optimal play should have 4 states."
        assert result[2][0].board.moves == ['x', 'o', ' ', 'x', ' ', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', ' ', 'x', 'x', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."
        assert result[2][2].board.moves == ['x', 'o', 'o', 'x', 'x', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."
        assert result[2][3].board.moves == ['x', 'o', 'o', 'x', 'x', 'x', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ', 'o']))
        test_state.player = 2
        result = fmin(test_state, -2, 2)
        assert isinstance(result, tuple) and len(
            result) == 3, "Result should be a 3-tuple"
        assert result[0] == - \
            1, "From ['x','o','x','x','x','o','o',' ','o'] min score should be -1."
        assert result[1] == 2, "From ['x','o','x','x','x','o','o',' ','o'] number of explored states should be 2."
        assert len(
            result[2]) == 2, "From ['x','o','x','x','x','o','o',' ','o'] optimal play should have 2 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', 'o',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['o', 'x', 'o', 'o', ' ', 'x', 'x', ' ', 'x']))
        test_state.player = 2
        result = fmin(test_state, -2, 2)
        assert result[0] == 0, "From ['o','x','o','o',' ','x','x',' ','x'] min score should be 0."
        assert result[1] == 5, "From ['o','x','o','o',' ','x','x',' ','x'] number of explored states should be 5."
        assert len(
            result[2]) == 3, "From ['o','x','o','o',' ','x','x',' ','x'] optimal play should have 3 states."
        assert result[2][0].board.moves == ['o', 'x', 'o', 'o', ' ', 'x', 'x', ' ',
                                            'x'], "From ['o','x','o','o',' ','x','x',' ','x'] incorrect optimal play."
        assert result[2][1].board.moves == ['o', 'x', 'o', 'o', ' ', 'x', 'x', 'o',
                                            'x'], "From ['o','x','o','o',' ','x','x',' ','x'] incorrect optimal play."
        assert result[2][2].board.moves == ['o', 'x', 'o', 'o', 'x', 'x', 'x', 'o',
                                            'x'], "From ['o','x','o','o',' ','x','x',' ','x'] incorrect optimal play."

        test_ok()

    @weight(25)
    def test_3_expectimax(self):
        fmax, fexpected, game_state, tic_tac_toe_board = get_locals(
            self.notebook_locals, ["maximize_score_expectimax", "expected_score_expectimax", "game_state", "tic_tac_toe_board"])
        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ', 'o']))
        result = fmax(test_state)
        assert isinstance(result, tuple) and len(
            result) == 3, "Result should be a 3-tuple"
        assert result[0] == 0, "From ['x','o','x','x','x','o','o',' ','o'] max score should be 0."
        assert result[1] == 2, "From ['x','o','x','x','x','o','o',' ','o'] number of explored states should be 2."
        assert len(
            result[2]) == 2, "From ['x','o','x','x','x','o','o',' ','o'] optimal play should have 2 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', ' ', 'o', 'o', ' ', 'o']))
        result = fmax(test_state)
        assert result[0] == 0, "From ['x','o','x','x',' ','o','o',' ','o'] max score should be 0."
        assert result[1] == 5, "From ['x','o','x','x',' ','o','o',' ','o'] number of explored states should be 5."
        assert len(
            result[2]) == 3, "From ['x','o','x','x',' ','o','o',' ','o'] optimal play should have 3 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', ' ', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', ' ', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."
        assert result[2][2].board.moves == ['x', 'o', 'x', 'x', 'o', 'o', 'o', 'x',
                                            'o'], "From ['x','o','x','x',' ','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', ' ', 'x', ' ', ' ', 'o', ' ', ' ']))
        result = fmax(test_state)
        assert result[0] == 1, "From ['x','o',' ','x',' ',' ','o',' ',' '] max score should be 1."
        assert result[1] == 246, "From ['x','o',' ','x',' ',' ','o',' ',' '] number of explored states should be 246."
        assert len(
            result[2]) == 4, "From ['x','o',' ','x',' ',' ','o',' ',' '] optimal play should have 4 states."
        assert result[2][0].board.moves == ['x', 'o', ' ', 'x', ' ', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', ' ', 'x', 'x', ' ', 'o', ' ',
                                            ' '], "From ['x','o',' ','x',' ',' ','o',' ',' '] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ', 'o']))
        test_state.player = 2
        result = fexpected(test_state)
        assert isinstance(result, tuple) and len(
            result) == 3, "Result should be a 3-tuple"
        assert result[0] == - \
            1, "From ['x','o','x','x','x','o','o',' ','o'] min score should be -1."
        assert result[1] == 2, "From ['x','o','x','x','x','o','o',' ','o'] number of explored states should be 2."
        assert len(
            result[2]) == 2, "From ['x','o','x','x','x','o','o',' ','o'] optimal play should have 2 states."
        assert result[2][0].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', ' ',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."
        assert result[2][1].board.moves == ['x', 'o', 'x', 'x', 'x', 'o', 'o', 'o',
                                            'o'], "From ['x','o','x','x','x','o','o',' ','o'] incorrect optimal play."

        test_state = game_state(tic_tac_toe_board(
            ['o', 'x', 'o', 'o', ' ', 'x', 'x', ' ', 'x']))
        test_state.player = 2
        result = fexpected(test_state)
        assert result[0] == 0.5, "From ['o','x','o','o',' ','x','x',' ','x'] min score should be 0.5."
        assert result[1] == 5, "From ['o','x','o','o',' ','x','x',' ','x'] number of explored states should be 5."
        assert len(
            result[2]) == 3, "From ['o','x','o','o',' ','x','x',' ','x'] optimal play should have 3 states."
        assert result[2][0].board.moves == ['o', 'x', 'o', 'o', ' ', 'x', 'x', ' ',
                                            'x'], "From ['o','x','o','o',' ','x','x',' ','x'] incorrect optimal play."

        test_ok()

    @weight(10)  # TODO: points for XA are still undecided
    @timeout_decorator.timeout(120.0)
    def test_xa_opponent_model(self):
        opponent_model, game_state, tic_tac_toe_board = get_locals(
            self.notebook_locals, ["opponent_model", "game_state", "tic_tac_toe_board"])
        policy = _make_student_policy(opponent_model, game_state, tic_tac_toe_board)
        games = _play_games(policy, _XA_NUM_GAMES, _XA_SEED)
        print(_format_games(games))
        tally = Counter(r for _, r in games)
        assert tally['win'] >= _XA_MIN_WINS, \
            f"Won {tally['win']}/{_XA_NUM_GAMES} games; need more than {_XA_MIN_WINS - 1}. See the games listed above."
        test_ok()

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_4_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Eomuktang".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
