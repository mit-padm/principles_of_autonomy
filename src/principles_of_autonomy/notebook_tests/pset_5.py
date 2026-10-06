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
# Problem 2A: modeling an imperfect opponent.
# The student plays o (second) against a hidden x opponent. Games are played
# with the notebook's game_state and tic_tac_toe_board classes; caches are keyed
# by the board as a tuple of 9 symbols ('x', 'o', ' '), row by row.
# ---------------------------------------------------------------------------
import random
from collections import Counter

_2A_NUM_GAMES = 100
_2A_SEED = 0
_2A_MIN_WINS = 21  # "more than 20/100"; the intended model averages ~35 wins

# Hidden opponent: always opens in corner 0, answers o's center with edge 5 or
# 7, then plays perfectly, except in the two (mirror-image) positions below,
# where with probability _2A_BLUNDER_PROB it makes a threat that forces o to
# block into a fork.
_2A_BLUNDER_PROB = 0.35
_2A_TRAPS = {
    ('x', ' ', ' ', ' ', 'o', 'x', ' ', ' ', 'o'): 3,
    ('x', ' ', ' ', ' ', 'o', ' ', ' ', 'x', 'o'): 1,
}


def _key(state):
    return tuple(state.board.moves)


def _cell_played(state, succ):
    """The cell that differs between a state and one of its successors."""
    return next(i for i, (a, b) in enumerate(zip(state.board.moves, succ.board.moves)) if a != b)


def _make_minimax():
    """Returns a cached function mapping a game_state to its MiniMax value."""
    cache = {}

    def minimax(state):
        k = _key(state)
        if k not in cache:
            if state.terminal_check():
                cache[k] = state.score()
            else:
                vals = [minimax(s) for s in state.successors()]
                cache[k] = max(vals) if state.player == 1 else min(vals)
        return cache[k]

    return minimax


def _make_opponent():
    """x's policy: maps a game_state (x to move) to the chosen successor."""
    minimax = _make_minimax()

    def move(state, rng):
        succs = state.successors()
        cells = [_cell_played(state, s) for s in succs]
        b = _key(state)
        if b.count(' ') == 9:
            return succs[cells.index(0)]
        if b.count(' ') == 7 and b[0] == 'x' and b[4] == 'o':
            return succs[cells.index(rng.choice([5, 7]))]
        if b in _2A_TRAPS and rng.random() < _2A_BLUNDER_PROB:
            return succs[cells.index(_2A_TRAPS[b])]
        vals = [minimax(s) for s in succs]
        return rng.choice([s for s, v in zip(succs, vals) if v == max(vals)])

    return move


def _make_student_policy(opponent_model, game_state, tic_tac_toe_board):
    """o's policy: expectimax where x nodes are weighted by the student's model
    and o minimizes. Ties go to the lowest-numbered cell (the first successor)."""
    model_cache, value_cache = {}, {}

    def model(state):
        b = _key(state)
        if b not in model_cache:
            # The student's model sees the board alone: no parent, player 1 (x to move).
            fresh = game_state(tic_tac_toe_board(list(b)))
            succs = fresh.successors()
            probs = list(opponent_model(fresh))
            board_str = list(b)
            assert len(probs) == len(succs), \
                f"opponent_model returned {len(probs)} probabilities for {board_str}, which has {len(succs)} successors."
            assert all(p >= 0 for p in probs), f"opponent_model returned a negative probability for {board_str}: {probs}"
            assert abs(sum(probs) - 1) < 1e-6, f"opponent_model probabilities for {board_str} sum to {sum(probs)}, not 1."
            model_cache[b] = list(zip(succs, probs))
        return model_cache[b]

    def value(state):
        b = _key(state)
        if b not in value_cache:
            if state.terminal_check():
                value_cache[b] = state.score()
            elif state.player == 1:
                value_cache[b] = sum(p * value(s) for s, p in model(state))
            else:
                value_cache[b] = min(value(s) for s in state.successors())
        return value_cache[b]

    def policy(state):
        succs = state.successors()
        vals = [value(s) for s in succs]
        best = min(vals)
        return next(s for s, v in zip(succs, vals) if v <= best + 1e-9)

    return policy


def _play_games(policy, game_state, tic_tac_toe_board, num_games, seed):
    """Returns a list of (moves, result) where moves are cell indices, x first."""
    rng = random.Random(seed)
    opponent = _make_opponent()
    games = []
    for _ in range(num_games):
        state, moves = game_state(tic_tac_toe_board([' '] * 9)), []
        while not state.terminal_check():
            succ = opponent(state, rng) if state.player == 1 else policy(state)
            moves.append(_cell_played(state, succ))
            state = succ
        result = {1: 'loss', 0: 'tie', -1: 'win'}[state.score()]
        games.append((tuple(moves), result))
    return games


def _format_moves(moves):
    return ' '.join(('X' if k % 2 == 0 else 'O') + str(i) for k, i in enumerate(moves))


def _format_games(games):
    """Every game in the order played, one per line, for pasting into an LLM."""
    num_wins = sum(result == 'win' for _, result in games)
    lines = [
        f"You (O) won {num_wins} of {len(games)} games (need more than {_2A_MIN_WINS - 1}).",
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

    @weight(35)
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

    @weight(5)
    @timeout_decorator.timeout(120.0)
    def test_4_opponent_model(self):
        opponent_model, game_state, tic_tac_toe_board = get_locals(
            self.notebook_locals, ["opponent_model", "game_state", "tic_tac_toe_board"])
        policy = _make_student_policy(opponent_model, game_state, tic_tac_toe_board)
        games = _play_games(policy, game_state, tic_tac_toe_board, _2A_NUM_GAMES, _2A_SEED)
        print(_format_games(games))
        tally = Counter(r for _, r in games)
        assert tally['win'] >= _2A_MIN_WINS, \
            f"Won {tally['win']}/{_2A_NUM_GAMES} games; need more than {_2A_MIN_WINS - 1}. See the games listed above."
        test_ok()

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_5_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Eomuktang".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
