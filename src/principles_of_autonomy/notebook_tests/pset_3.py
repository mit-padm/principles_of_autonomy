import unittest
from copy import deepcopy
from itertools import product
import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
# from nose.tools import assert_equal

from principles_of_autonomy.grader import get_locals
import numpy as np
import sympy

# helper functions

# Function for tests
def test_ok():
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Test passed!!</strong>
        </div>""", raw=True)
    except:
        print("Test passed!!")


_PACMAN_DIRECTIONS = {
    "North": (-1, 0),
    "South": (1, 0),
    "East": (0, 1),
    "West": (0, -1),
}


def _pacman_worlds(grid, actions, percepts, start):
    """Enumerate possible trajectories on the known map without a KB encoding."""
    assert grid and grid[0] and all(
        len(row) == len(grid[0]) and all(cell in ("#", ".") for cell in row)
        for row in grid
    ), "Each oracle case needs a nonempty rectangular grid containing only '#' and '.'."
    assert len(percepts) == len(actions) + 1 and all(
        isinstance(reading, tuple) and len(reading) == 4
        and all(type(bit) is bool for bit in reading) for reading in percepts
    ), "Each oracle case needs a four-Boolean percept tuple initially and after every action."
    rows, cols = len(grid), len(grid[0])
    cells = list(product(range(rows), range(cols)))
    wall_symbols = {cell: sympy.Symbol(f"W_{cell[0]}_{cell[1]}") for cell in cells}
    position_symbols = {
        (row, col, t): sympy.Symbol(f"P_{row}_{col}_{t}")
        for row, col in cells for t in range(len(actions) + 1)
    }
    symbols = list(wall_symbols.values()) + list(position_symbols.values())
    worlds = []
    walls = {cell: grid[cell[0]][cell[1]] == "#" for cell in cells}

    def blocked(row, col):
        return not (0 <= row < rows and 0 <= col < cols) or walls[(row, col)]

    for initial in cells if start is None else [start]:
        if walls[initial]:
            continue
        trajectory = [initial]
        for action in actions:
            row, col = trajectory[-1]
            dr, dc = _PACMAN_DIRECTIONS[action]
            target = (row + dr, col + dc)
            trajectory.append(trajectory[-1] if blocked(*target) else target)
        readings = [
            tuple(blocked(row + dr, col + dc)
                  for dr, dc in _PACMAN_DIRECTIONS.values())
            for row, col in trajectory
        ]
        if any(observed != actual
               for observed, actual in zip(percepts, readings)):
            continue
        world = {wall_symbols[cell]: walls[cell] for cell in cells}
        world.update({symbol: trajectory[t] == (row, col)
                      for (row, col, t), symbol in position_symbols.items()})
        worlds.append(world)
    return symbols, worlds


def _pacman_satisfiable(formula):
    """Use internal Tseitin gates so equivalent DNF answers stay inexpensive.

    The gates are private to this SAT query; students may use only W/P symbols.
    SymPy's default CNF conversion distributes a large DNF exponentially.
    """
    clauses, encoded = [], {}

    def literal(expr):
        if expr in (sympy.true, sympy.false) or isinstance(expr, sympy.Symbol):
            return expr
        if expr in encoded:
            return encoded[expr]
        if expr.func is sympy.Not:
            result = ~literal(expr.args[0])
        elif expr.func in (sympy.And, sympy.Or):
            args = [literal(arg) for arg in expr.args]
            result = sympy.Dummy("pacman_gate")
            if expr.func is sympy.And:
                clauses.extend(sympy.Or(~result, arg) for arg in args)
                clauses.append(sympy.Or(result, *(~arg for arg in args)))
            else:
                clauses.extend(sympy.Or(result, ~arg) for arg in args)
                clauses.append(sympy.Or(~result, *args))
        else:
            result = literal(sympy.to_nnf(expr, simplify=False))
        encoded[expr] = result
        return result

    root = literal(formula)
    return sympy.satisfiable(sympy.And(root, *clauses), algorithm="dpll2")


def _pacman_counterexample(kind, grid, actions, percepts, start, symbols, witness):
    # SAT models can omit irrelevant variables. False completes those variables
    # deterministically, yielding a full, reproducible assignment over the API.
    assignment = {symbol: bool(witness.get(symbol, False)) for symbol in symbols}
    walls = ["".join("#" if assignment[sympy.Symbol(f"W_{row}_{col}")] else "."
                     for col in range(len(grid[0]))) for row in range(len(grid))]
    positions = {
        t: [(row, col) for row in range(len(grid)) for col in range(len(grid[0]))
            if assignment[sympy.Symbol(f"P_{row}_{col}_{t}")]]
        for t in range(len(actions) + 1)
    }
    named_assignment = {str(symbol): assignment[symbol]
                        for symbol in sorted(symbols, key=str)}
    expected = kind == "excludes an admissible world"
    return (
        f"Your KB {kind}.\n"
        f"Expected KB value: {expected}\n"
        f"Actual KB value: {not expected}\n"
        f"Inputs: grid={grid!r}, actions={actions!r}, percepts={percepts!r}, start={start!r}\n"
        "Wall map (#=wall, .=open):\n" + "\n".join(walls) + "\n"
        f"Positions by timestep: {positions!r}\n"
        "Each list includes every position assigned True, even zero or multiple positions.\n"
        f"Complete assignment: {named_assignment!r}\n"
        "Use this counterexample to revise your KB, then rerun the tests."
    )


def _check_pacman_kb(build_pacman_kb, grid, actions, percepts, start=None):
    """Compare exactly the admissible worlds, without comparing formula syntax."""
    symbols, worlds = _pacman_worlds(grid, actions, percepts, start)
    # Keep the oracle and its diagnostics independent of student input mutation.
    formula = build_pacman_kb(deepcopy(grid), list(actions), deepcopy(percepts), start)
    assert isinstance(formula, sympy.logic.boolalg.Boolean), (
        "build_pacman_kb must return a SymPy Boolean formula (including sympy.true/false)."
    )
    unknown = formula.free_symbols.difference(symbols)
    assert not unknown, (
        "The KB uses undeclared symbols: " + ", ".join(sorted(map(str, unknown)))
        + ". Use only W_row_col and P_row_col_t with the documented indices."
    )
    for node in sympy.preorder_traversal(formula):
        assert (isinstance(node, (sympy.Symbol, sympy.logic.boolalg.BooleanFunction))
                or node is sympy.true or node is sympy.false), (
            "The KB must be propositional: use Boolean connectives over W/P symbols, "
            f"not arithmetic, relations, or other atoms (found {node!r})."
        )

    # This is the existential check expected & ~formula, evaluated one complete
    # admissible assignment at a time to avoid distributing a large DNF to CNF.
    for world in worlds:
        assert formula.xreplace(world) == sympy.true, _pacman_counterexample(
            "excludes an admissible world", grid, actions, percepts, start, symbols, world)

    expected = sympy.Or(*(sympy.And(*(symbol if value else ~symbol
                                     for symbol, value in world.items()))
                          for world in worlds))
    witness = _pacman_satisfiable(sympy.And(formula, ~expected))
    assert witness is False, _pacman_counterexample(
        "admits an impossible world", grid, actions, percepts, start, symbols,
        witness if witness is not False else {})


class TestPSet3(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    def test_01(self):
        q1_answer = get_locals(self.notebook_locals, ["q1_answer"])
        answer = 1
        assert q1_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_02(self):
        q2_answer = get_locals(self.notebook_locals, ["q2_answer"])
        answer = (True, False, True, True, True, False, True, False)
        assert len(q2_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q2_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_03(self):
        q3_answer = get_locals(self.notebook_locals, ["q3_answer"])
        answer = (True, False, True, True, True, True, True, True)
        assert len(q3_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q3_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_04(self):
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        answer = (False, True, False)
        assert len(q4_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q4_answer == answer, "Incorrect values."

        test_ok()

    @weight(10)
    def test_05(self):
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        answer =  ('F', 'V', 'T', 'F', 'F', 'T', 'U')
        assert len(q5_answer) == len(answer), f"Incorrect number of values, need {len(answer)} characters"
        assert q5_answer == answer, "Incorrect values."

        test_ok()


    @weight(5)
    def test_06(self):
        warmup = get_locals(self.notebook_locals, 
                                                  ["warmup"])
        answer = warmup()
        assert len(answer) == 3, "Incorrect number of clauses"
        assert len(answer[0])==3 and len(answer[1])==3 and len(answer[2])==2, "Incorrect number of literals in each clause"
        assert (4 in answer[0] and -5 in answer[0] and -6 in answer[0] and 
                6 in answer[1] and 5 in answer[1] and -1 in answer[1] and
                2 in answer[2] and 3 in answer[2]), "Incorrect literals"
        assert answer == [[4, -5, -6], [6, 5, -1], [2, 3]], "Incorrect order of literals"

        test_ok()

    @weight(23)
    def test_07(self):
        run_inference_dpll = get_locals(self.notebook_locals,
                                           ["run_inference_dpll"])
        assert run_inference_dpll([[-1, 2]]) in [(True, {1: True, 2: True}), (True, {1: False, 2: True}), (True, {1: False, 2: False})]
        assert run_inference_dpll([[1], [-1]]) == (False, None)
        assert run_inference_dpll([[1, 2, 3], [-1, -2, -3], [1, -2, 3], [-1], [-3]]) == (False, None)
        assert run_inference_dpll([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32]]) == (True, {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True, 16: True, 17: True, 18: True, 19: True, 20: True, 21: True, 22: True, 23: True, 24: True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True})

        test_ok()

    @weight(5)
    def test_08(self):
        formula1_is_satisfiable = get_locals(self.notebook_locals, ["formula1_is_satisfiable"])
        is_satisfiable, formula = formula1_is_satisfiable()
        assert is_satisfiable == True, "Incorrect boolean value"
        assert isinstance(formula, sympy.logic.boolalg.Boolean), "Not a sympy formula"

        test_ok()

    @weight(5)
    def test_09(self):
        formula2_is_satisfiable = get_locals(self.notebook_locals, ["formula2_is_satisfiable"])
        is_satisfiable, formula = formula2_is_satisfiable()
        assert is_satisfiable == False, "Incorrect boolean value"
        assert isinstance(formula, sympy.logic.boolalg.Boolean), "Not a sympy formula"

        test_ok()

    @weight(19)
    def test_10(self):
        infer_unknown_values = get_locals(self.notebook_locals, ["infer_unknown_values"])
        assert infer_unknown_values([["U", "C", "C"], ["S", "C", "U"], ["U", "U", "C"]]) == [["C", "C", "C"], ["S", "C", "C"], ["F", "S", "C"]]
        assert infer_unknown_values([["U", "S", "C", "U"], ["U", "U", "C", "U"], ["U", "S", "C", "U"]]) == [["F", "S", "C", "C"], ["S", "C", "C", "C"], ["F", "S", "C", "C"]]
        assert infer_unknown_values([["U", "U", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "F", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"]]) == [["C", "C", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "S", "U", "U", "U", "U"], ["U", "U", "S", "F", "S", "U", "U", "U"], ["U", "U", "U", "S", "U", "U", "U", "U"]]

        test_ok()

    # Pac-Man (AI-enabled): 2 points per test.
    @weight(2)
    @timeout_decorator.timeout(20.0)
    def test_11_pacman_worlds(self):
        """test_11_pacman_worlds: Known map, uncertain initial position, unique positions, and occupancy."""
        build = get_locals(self.notebook_locals, ["build_pacman_kb"])
        for grid, reading, start in [
            ([["."]], (True, True, True, True), None),
            ([[".", "#", ".", "#", "."]], (True, True, True, True), None),
            ([[".", ".", ".", "#", "."]], (True, True, False, True), (0, 0)),
            ([[".", ".", ".", "."]], (True, True, False, False), None),
            ([["#", "#"]], (True, True, True, True), None),
            ([["#", "."]], (True, True, False, True), (0, 0)),
        ]:
            _check_pacman_kb(build, grid, [], [reading], start)
        test_ok()

    @weight(2)
    @timeout_decorator.timeout(20.0)
    def test_12_pacman_sensors(self):
        """test_12_pacman_sensors: Perfect N/S/E/W readings, including false bits and boundaries."""
        build = get_locals(self.notebook_locals, ["build_pacman_kb"])
        cases = [
            ([["."]], (True, True, True, True), None),
            ([["."]], (False, True, True, True), None),
            ([[".", ".", "#"], [".", ".", "."]],
             (True, False, False, True), (0, 0)),
            ([[".", "#", "."], [".", ".", "."]],
             (False, True, False, True), None),
            ([[".", ".", "."], [".", ".", "."], [".", ".", "."]],
             (False, False, False, False), None),
        ]
        for grid, reading, start in cases:
            _check_pacman_kb(build, grid, [], [reading], start)
        test_ok()

    @weight(2)
    @timeout_decorator.timeout(20.0)
    def test_13_pacman_actions(self):
        """test_13_pacman_actions: Each cardinal direction; both open moves and blocked attempts."""
        build = get_locals(self.notebook_locals, ["build_pacman_kb"])
        for action in _PACMAN_DIRECTIONS:
            # Repeated local readings leave several positions indistinguishable
            # to sensors, so these cases still require correct movement rules.
            vertical = action in ("North", "South")
            grid = [["."]] * 5 if vertical else [["."] * 5]
            start = (2, 0) if vertical else (0, 2)
            reading = (False, False, True, True) if vertical else (True, True, False, False)
            _check_pacman_kb(build, grid, [action], [reading, reading], start)
        _check_pacman_kb(build, [["."]], ["North", "West", "South"],
                         [(True, True, True, True)] * 4, (0, 0))
        _check_pacman_kb(build, [[".", "#", ".", "#", "."]], ["East"],
                         [(True, True, True, True)] * 2, (0, 0))
        _check_pacman_kb(build, [[".", "#"], [".", "."]],
                         ["East", "South", "East"],
                         [(True, False, True, True), (True, False, True, True),
                          (False, True, False, True), (True, True, True, False)], (0, 0))
        test_ok()

    @weight(2)
    @timeout_decorator.timeout(20.0)
    def test_14_pacman_histories(self):
        """test_14_pacman_histories: Persistent maps, unknown starts, observation timing, contradictions."""
        build = get_locals(self.notebook_locals, ["build_pacman_kb"])
        grid = [[".", ".", "#"], [".", ".", "."], ["#", ".", "."]]
        _check_pacman_kb(build, grid, ["East", "South", "West"],
                         [(True, False, False, True), (True, False, True, False),
                          (False, False, False, False), (False, True, False, True)], (0, 0))
        _check_pacman_kb(build, [["."] * 6], ["East", "West", "East"],
                         [(True, True, False, False)] * 4)
        _check_pacman_kb(build, [[".", ".", "."]], ["East"],
                         [(True, True, False, True), (True, True, False, False)], (0, 0))
        _check_pacman_kb(build, [[".", "#", "."]], ["North"],
                         [(True, True, True, True), (True, True, False, True)], (0, 0))
        _check_pacman_kb(build, [[".", "."]], ["East"],
                         [(True, True, False, True), (True, True, True, False)], (0, 0))
        _check_pacman_kb(build, [[".", "."]], ["East"],
                         [(True, True, False, True), (True, True, False, True)], (0, 0))
        test_ok()

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_15_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Optimus Prime".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
