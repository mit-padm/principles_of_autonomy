"""Instructor checks for the Pacman problem and its semantic autograder.

From the repository root, with the assignment dependencies installed, run::

    PYTHONPATH=principles_of_autonomy/src python -m unittest discover \
        -s principles_of_autonomy/tests -p 'test_pset_3_pacman.py'

Only the notebook's Pacman helper and reference-builder cells are executed.
These tests check the grader against deliberately incorrect encodings, as well
as the reference solution and a logically equivalent alternative encoding.
Notebook-dependent tests skip when this package is cloned without the parent
assignment repository.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import sympy

from principles_of_autonomy.notebook_tests import pset_3
from principles_of_autonomy.grader import Grader


NOTEBOOK = (Path(__file__).resolve().parents[2] / "ps3-logic" /
            "ProblemSet03_Logic.ipynb")
PACMAN_TESTS = (
    "test_11_pacman_worlds",
    "test_12_pacman_sensors",
    "test_13_pacman_actions",
    "test_14_pacman_histories",
)


def load_reference_builder():
    if not NOTEBOOK.is_file():
        raise unittest.SkipTest("Reference notebook requires the parent assignment repository.")
    namespace = {"sympy": sympy}
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        source = cell["source"]
        source = source if isinstance(source, str) else "".join(source)
        if cell["cell_type"] == "code" and (
                "def pacman_wall(" in source or
                "def build_pacman_kb(" in source):
            exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)
    return namespace["build_pacman_kb"]


class PacmanSatEncodingTests(unittest.TestCase):
    def test_nested_connectives_match_direct_sympy_sat(self):
        a, b, c = sympy.symbols("A B C")
        formulas = [
            sympy.Or(~(a | b), c),
            sympy.Implies(a & b, b | c),
            sympy.Equivalent(a, b | c),
            sympy.Xor(a, b & c),
            sympy.ITE(a, b | c, ~b & ~c),
            sympy.And(sympy.Equivalent(a, b), sympy.Xor(a, b)),
            sympy.And(a | b, ~a, ~b),
            sympy.true,
            sympy.false,
        ]
        for formula in formulas:
            with self.subTest(formula=formula):
                expected = sympy.satisfiable(formula)
                actual = pset_3._pacman_satisfiable(formula)
                self.assertEqual(actual is False, expected is False)
                if actual is not False:
                    assignment = {atom: bool(actual.get(atom, False))
                                  for atom in formula.free_symbols}
                    self.assertEqual(formula.xreplace(assignment), sympy.true)


class PacmanGraderRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = staticmethod(load_reference_builder())

    def check(self, builder, grid, actions, percepts, start=None):
        pset_3._check_pacman_kb(builder, grid, actions, percepts, start)

    def reject(self, builder, grid, actions, percepts, start=None, *, semantic=True):
        with self.assertRaises(AssertionError) as failure:
            self.check(builder, grid, actions, percepts, start)
        message = str(failure.exception)
        if semantic:
            self.assertIn("Your KB", message,
                          "A semantic mutant must fail the world comparison.")
        return message

    def test_reference_passes_all_public_grading_groups(self):
        for name in PACMAN_TESTS:
            with self.subTest(grading_group=name):
                case = pset_3.TestPSet3(name, {"build_pacman_kb": self.build})
                getattr(case, name)()

    def test_reference_preserves_input_data(self):
        inputs = ([[".", "."], ["#", "."]], ["East", "South"],
                  [(True, True, False, True), (True, False, True, False),
                   (False, True, True, True)])
        original = copy.deepcopy(inputs)
        self.check(self.build, *inputs, start=(0, 0))
        self.assertEqual(inputs, original)

    def test_equivalent_dnf_is_accepted(self):
        def dnf_builder(*args):
            return sympy.to_dnf(self.build(*args), simplify=True, force=True)

        self.check(dnf_builder, [["."] * 4], [], [(True, True, False, False)])

    def test_false_is_accepted_for_an_impossible_history(self):
        self.check(lambda *args: sympy.false, [["."]], [],
                   [(False, True, True, True)], (0, 0))

    def test_false_is_accepted_when_every_cell_is_a_wall(self):
        self.check(lambda *args: sympy.false, [["#"]], [], [(True,) * 4])

    def test_constants_fail_on_a_satisfiable_problem(self):
        for constant in (sympy.true, sympy.false):
            with self.subTest(formula=constant):
                message = self.reject(lambda *args: constant,
                                      [["."]], [], [(True,) * 4], (0, 0))
                expected = ("admits an impossible world" if constant is sympy.true
                            else "excludes an admissible world")
                self.assertIn(expected, message)
                self.assertIn(f"Expected KB value: {constant is sympy.false}", message)
                self.assertIn(f"Actual KB value: {constant is sympy.true}", message)
                for detail in ("Inputs:", "Wall map (#=wall, .=open):",
                               "Positions by timestep:", "Complete assignment:",
                               "W_0_0", "P_0_0_0"):
                    self.assertIn(detail, message)

    def test_grading_json_retains_failing_test_and_counterexample(self):
        class FeedbackCase(unittest.TestCase):
            def __init__(self, test_name, notebook_locals):
                super().__init__(test_name)
                self.notebook_locals = notebook_locals

            test_11_pacman_worlds = pset_3.TestPSet3.test_11_pacman_worlds

        for constant in (sympy.true, sympy.false):
            with self.subTest(formula=constant), tempfile.TemporaryDirectory() as directory:
                results_path = Path(directory) / "results.json"
                Grader.grade_output([FeedbackCase],
                                    [{"build_pacman_kb": lambda *args: constant}],
                                    results_path)
                results = json.loads(results_path.read_text())
                self.assertEqual(len(results["tests"]), 1)
                failure = results["tests"][0]
                self.assertIn("test_11_pacman_worlds", failure["name"])
                self.assertEqual(failure["status"], "failed")
                for detail in ("Inputs: grid=", "actions=", "percepts=", "start=",
                               "Wall map (#=wall, .=open):", "Positions by timestep:",
                               "Complete assignment:",
                               f"Expected KB value: {constant is sympy.false}",
                               f"Actual KB value: {constant is sympy.true}"):
                    self.assertIn(detail, failure["output"])

    def test_unknown_start_cannot_be_fixed_arbitrarily(self):
        def fix_start(grid, actions, percepts, start):
            return self.build(grid, actions, percepts,
                              (0, 1) if start is None else start)

        self.reject(fix_start, [["."] * 4], [], [(True, True, False, False)])

    def test_known_map_facts_cannot_be_omitted(self):
        w0, w1 = sympy.symbols("W_0_0 W_0_1")
        p0, p1, p2 = sympy.symbols("P_0_0_0 P_0_1_0 P_0_2_0")
        # The wall at column 2 is not sensed from column 0, but its known
        # map fact must still hold in every admitted world.
        formula = sympy.And(~w0, ~w1, p0, ~p1, ~p2)
        self.reject(lambda *args: formula, [[".", ".", "#"]], [],
                    [(True, True, False, True)], (0, 0))

    def test_a_single_satisfying_world_is_not_a_complete_kb(self):
        def one_world(*args):
            formula = self.build(*args)
            model = sympy.satisfiable(formula)
            grid, actions = args[:2]
            atoms = [sympy.Symbol(f"W_{row}_{col}")
                     for row in range(len(grid)) for col in range(len(grid[0]))]
            atoms += [sympy.Symbol(f"P_{row}_{col}_{time}")
                      for time in range(len(actions) + 1)
                      for row in range(len(grid)) for col in range(len(grid[0]))]
            return sympy.And(formula, *[
                atom if model.get(atom, False) else ~atom
                for atom in atoms])

        self.reject(one_world, [["."] * 4], [],
                    [(True, True, False, False)])

    def test_at_least_one_position_is_required(self):
        walls = sympy.symbols("W_0_0 W_0_1 W_0_2 W_0_3")
        p0, p1, p2, p3 = sympy.symbols("P_0_0_0 P_0_1_0 P_0_2_0 P_0_3_0")
        formula = sympy.And(*(~wall for wall in walls), ~p0, ~p3, ~(p1 & p2))
        self.reject(lambda *args: formula, [["."] * 4], [],
                    [(True, True, False, False)])

    def test_at_most_one_position_is_required(self):
        walls = sympy.symbols("W_0_0 W_0_1 W_0_2 W_0_3")
        p0, p1, p2, p3 = sympy.symbols("P_0_0_0 P_0_1_0 P_0_2_0 P_0_3_0")
        formula = sympy.And(*(~wall for wall in walls), ~p0, ~p3, p1 | p2)
        self.reject(lambda *args: formula, [["."] * 4], [],
                    [(True, True, False, False)])

    def test_pacman_cannot_occupy_a_wall(self):
        walls = sympy.symbols("W_0_0 W_0_1 W_0_2 W_0_3 W_0_4")
        p0, p1, p2, p3, p4 = sympy.symbols(
            "P_0_0_0 P_0_1_0 P_0_2_0 P_0_3_0 P_0_4_0")
        # The wall at column 1 and the open cell at column 3 have identical
        # neighboring-wall readings; only occupancy rules distinguish them.
        formula = sympy.And(*(wall if col == 1 else ~wall
                              for col, wall in enumerate(walls)),
                            ~p0, ~p2, ~p4, sympy.Xor(p1, p3))
        self.reject(lambda *args: formula, [[".", "#", ".", ".", "."]], [],
                    [(True, True, False, False)])

    def test_percepts_cannot_be_ignored(self):
        w0, w1, p0, p1 = sympy.symbols("W_0_0 W_0_1 P_0_0_0 P_0_1_0")
        # Map and occupancy hold, but the reading localizes Pac-Man to column
        # 0. Omitting sensing incorrectly leaves column 1 possible as well.
        formula = sympy.And(~w0, ~w1, sympy.Xor(p0, p1))
        self.reject(lambda *args: formula, [[".", "."]], [],
                    [(True, True, False, True)])

    def test_sensor_order_is_north_south_east_west(self):
        def swap_north_south(grid, actions, percepts, start):
            changed = [(reading[1], reading[0], reading[2], reading[3])
                       for reading in percepts]
            return self.build(grid, actions, changed, start)

        self.reject(swap_north_south, [["."], ["."]], [],
                    [(True, False, True, True)], (0, 0))

    def test_final_percept_is_not_discarded(self):
        def ignore_final(grid, actions, percepts, start):
            # Both interior starts fit the initial reading. The final reading
            # distinguishes their trajectories, which this mutant combines.
            return sympy.Or(
                self.build(grid, actions,
                           [percepts[0], (True, True, False, False)], start),
                self.build(grid, actions,
                           [percepts[0], (True, True, True, False)], start))

        self.reject(ignore_final, [["."] * 4], ["East"],
                    [(True, True, False, False), (True, True, True, False)])

    def test_sensor_history_shares_one_static_map(self):
        # The east wall cannot be both present and absent while Pacman remains
        # in place after attempting to move beyond the northern boundary.
        inputs = ([[".", "#"]], ["North"],
                  [(True, True, True, True), (True, True, False, True)], (0, 0))
        self.check(self.build, *inputs)
        self.assertIs(sympy.satisfiable(self.build(*inputs)), False)

    def test_percepts_are_associated_with_the_correct_timestep(self):
        def reverse_readings(grid, actions, percepts, start):
            return self.build(grid, actions, list(reversed(percepts)), start)

        self.reject(reverse_readings, [[".", "."]], ["East"],
                    [(True, True, False, True), (True, True, True, False)], (0, 0))

    def test_unrelated_positions_cannot_replace_successor_axioms(self):
        def independent_times(grid, actions, percepts, start):
            clauses = []
            for time, reading in enumerate(percepts):
                step = self.build(grid, [], [reading], start if time == 0 else None)
                rename = {
                    sympy.Symbol(f"P_{row}_{col}_0"):
                    sympy.Symbol(f"P_{row}_{col}_{time}")
                    for row in range(len(grid)) for col in range(len(grid[0]))}
                clauses.append(step.xreplace(rename))
            return sympy.And(*clauses)

        self.reject(independent_times, [["."] * 5], ["East"],
                    [(True, True, False, False)] * 2, (0, 1))

    def test_a_blocked_move_must_allow_staying_in_place(self):
        def forbid_staying(grid, actions, percepts, start):
            moves = [sympy.Implies(sympy.Symbol(f"P_{row}_{col}_{time}"),
                                  ~sympy.Symbol(f"P_{row}_{col}_{time + 1}"))
                     for time in range(len(actions))
                     for row in range(len(grid)) for col in range(len(grid[0]))]
            return sympy.And(self.build(grid, actions, percepts, start), *moves)

        self.reject(forbid_staying, [[".", "#"]], ["East"],
                    [(True,) * 4] * 2, (0, 0))

    def test_invalid_formula_types_and_symbols_are_rejected(self):
        invalid = ({"W_0_0": False}, sympy.Integer(2),
                   sympy.Symbol("Ghost"), sympy.Eq(sympy.Symbol("W_0_0"), 1))
        for formula in invalid:
            with self.subTest(formula=formula):
                self.reject(lambda *args: formula, [["."]], [],
                            [(True,) * 4], (0, 0), semantic=False)


if __name__ == "__main__":
    unittest.main()
