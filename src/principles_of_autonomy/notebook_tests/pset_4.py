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

# helper functions
# This uses TYPING
# BLOCKS_DOMAIN = """(define (domain blocks)
#     (:requirements :strips :typing)
#     (:types block)
#     (:predicates
#         (on ?x - block ?y - block)
#         (ontable ?x - block)
#         (clear ?x - block)
#         (handempty)
#         (holding ?x - block)
#     )

#     (:action pick-up
#         :parameters (?x - block)
#         :precondition (and
#             (clear ?x)
#             (ontable ?x)
#             (handempty)
#         )
#         :effect (and
#             (not (ontable ?x))
#             (not (clear ?x))
#             (not (handempty))
#             (holding ?x)
#         )
#     )

#     (:action put-down
#         :parameters (?x - block)
#         :precondition (and
#             (holding ?x)
#         )
#         :effect (and
#             (not (holding ?x))
#             (clear ?x)
#             (handempty)
#             (ontable ?x))
#         )

#     (:action stack
#         :parameters (?x - block ?y - block)
#         :precondition (and
#             (holding ?x)
#             (clear ?y)
#         )
#         :effect (and
#             (not (holding ?x))
#             (not (clear ?y))
#             (clear ?x)
#             (handempty)
#             (on ?x ?y)
#         )
#     )

#     (:action unstack
#         :parameters (?x - block ?y - block)
#         :precondition (and
#             (on ?x ?y)
#             (clear ?x)
#             (handempty)
#         )
#         :effect (and
#             (holding ?x)
#             (clear ?y)
#             (not (clear ?x))
#             (not (handempty))
#             (not (on ?x ?y))
#         )
#     )
# )
# """

# # This uses TYPING
# BLOCKS_PROBLEM = """(define (problem blocks)
#     (:domain blocks)
#     (:objects
#         d - block
#         b - block
#         a - block
#         c - block
#     )
#     (:init
#         (clear a)
#         (on a b)
#         (on b c)
#         (on c d)
#         (ontable d)
#         (handempty)
#     )
#     (:goal (and (on d c) (on c b) (on b a)))
# )
# """

# # The BW domain does not use TYPING
# BW_BLOCKS_DOMAIN = """(define (domain prodigy-bw)
#   (:requirements :strips)
#   (:predicates (on ?x ?y)
#                (ontable ?x)
#                (clear ?x)
#                (handempty)
#                (holding ?x)
#                )
#   (:action pick-up
#              :parameters (?ob1)
#              :precondition (and (clear ?ob1) (ontable ?ob1) (handempty))
#              :effect
#              (and (not (ontable ?ob1))
#                    (not (clear ?ob1))
#                    (not (handempty))
#                    (holding ?ob1)))
#   (:action put-down
#              :parameters (?ob)
#              :precondition (holding ?ob)
#              :effect
#              (and (not (holding ?ob))
#                    (clear ?ob)
#                    (handempty)
#                    (ontable ?ob)))
#   (:action stack
#              :parameters (?sob ?sunderob)
#              :precondition (and (holding ?sob) (clear ?sunderob))
#              :effect
#              (and (not (holding ?sob))
#                    (not (clear ?sunderob))
#                    (clear ?sob)
#                    (handempty)
#                    (on ?sob ?sunderob)))
#   (:action unstack
#              :parameters (?sob ?sunderob)
#              :precondition (and (on ?sob ?sunderob) (clear ?sob) (handempty))
#              :effect
#              (and (holding ?sob)
#                    (clear ?sunderob)
#                    (not (clear ?sob))
#                    (not (handempty))
#                    (not (on ?sob ?sunderob)))))
# """


def get_task_definition_str(domain_pddl_str, problem_pddl_str):
    """Get Pyperplan task definition from PDDL domain and problem.

    This function is a lightweight wrapper around Pyperplan.

    Args:
      domain_pddl_str: A str, the contents of a domain.pddl file.
      problem_pddl_str: A str, the contents of a problem.pddl file.

    Returns:
      task: a structure defining the problem
    """
    # Parsing the PDDL
    domain_file = tempfile.NamedTemporaryFile(delete=False)
    problem_file = tempfile.NamedTemporaryFile(delete=False)
    with open(domain_file.name, 'w') as f:
        f.write(domain_pddl_str)
    with open(problem_file.name, 'w') as f:
        f.write(problem_pddl_str)
    parser = Parser(domain_file.name, problem_file.name)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    os.remove(domain_file.name)
    os.remove(problem_file.name)

    # Ground the PDDL
    task = grounding.ground(problem)
    return task


def run_planning(domain_pddl_str,
                 problem_pddl_str,
                 search_alg_name,
                 heuristic_name=None,
                 return_time=False):
    """Plan a sequence of actions to solve the given PDDL problem.

    This function is a lightweight wrapper around pyperplan.

    Args:
      domain_pddl_str: A str, the contents of a domain.pddl file.
      problem_pddl_str: A str, the contents of a problem.pddl file.
      search_alg_name: A str, the name of a search algorithm in
        pyperplan. Options: astar, wastar, gbf, bfs, ehs, ids, sat.
      heuristic_name: A str, the name of a heuristic in pyperplan.
        Options: blind, hadd, hmax, hsa, hff, lmcut, landmark.
      return_time:  Bool. Set to `True` to return the planning time.

    Returns:
      plan: A list of actions; each action is a pyperplan Operator.
    """
    # Ground the PDDL
    task = get_task_definition_str(domain_pddl_str, problem_pddl_str)

    # Get the search alg
    search_alg = planner.SEARCHES[search_alg_name]

    if heuristic_name is None:
        if not return_time:
            return search_alg(task)
        start_time = time.time()
        plan = search_alg(task)
        plan_time = time.time() - start_time
        return plan, plan_time

    # Get the heuristic
    heuristic = planner.HEURISTICS[heuristic_name](task)

    # Run planning
    start_time = time.time()
    plan = search_alg(task, heuristic)
    plan_time = time.time() - start_time

    if return_time:
        return plan, plan_time
    return plan

# ---------- Sudoku helpers (Section 2) ----------

# Thresholds for the adversarial Sudoku in 2B. The notebook asks for a puzzle with
# a unique solution and at least 40 clues on which row-major backtracking visits
# at least 1,000,000 nodes while row-major forward checking visits at most 20,000.
SUDOKU_MIN_CLUES = 40
SUDOKU_BT_MIN_STEPS = 1_000_000
SUDOKU_FC_MAX_STEPS = 20_000

# Both puzzles have unique solutions, so any valid completion is the answer.
SUDOKU_EASY = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
SUDOKU_MEDIUM = "200080300060070084030500209000105408000000000402706000301007040720040060004010003"
# Two 5s in row 0: no solution.
SUDOKU_CONFLICTING = "550000000" + "0" * 72


def _sudoku_peers(i):
    r, c = divmod(i, 9)
    return [j for j in range(81) if j != i and (
        j // 9 == r or j % 9 == c or (j // 27 == i // 27 and (j % 9) // 3 == c // 3))]


_SUDOKU_PEERS = [_sudoku_peers(i) for i in range(81)]


def _flatten_sudoku(grid):
    """Validate a 9x9 grid of ints in 0..9 and return it as a flat list of 81 ints."""
    rows = [list(row) for row in grid]
    assert len(rows) == 9 and all(len(row) == 9 for row in rows), \
        "A Sudoku grid must be a 9x9 array (9 rows of 9 entries)."
    flat = []
    for row in rows:
        for x in row:
            assert int(x) == x and 0 <= int(x) <= 9, \
                "Every Sudoku entry must be an integer from 0 (empty) to 9, found %r." % (x,)
            flat.append(int(x))
    return flat


def _as_grid(s):
    return [[int(s[9 * r + c]) for c in range(9)] for r in range(9)]


class _StepCapReached(Exception):
    pass


def _reference_backtracking_steps(flat, cap):
    """Row-major backtracking on the Sudoku CSP, counting nodes like 1B.

    Returns the number of nodes visited, or cap if the search is cut off there.
    """
    domains = [[x] if x else list(range(1, 10)) for x in flat]
    val = [0] * 81
    steps = [0]

    def dfs(k):
        steps[0] += 1
        if steps[0] >= cap:
            raise _StepCapReached
        if k == 81:
            return True
        for x in domains[k]:
            if all(val[p] != x for p in _SUDOKU_PEERS[k]):
                val[k] = x
                if dfs(k + 1):
                    return True
                val[k] = 0
        return False

    try:
        dfs(0)
    except _StepCapReached:
        return cap
    return steps[0]


def _reference_forward_checking_steps(flat, cap):
    """Row-major forward checking on the Sudoku CSP, counting nodes like 1C.

    Returns the number of nodes visited, or cap if the search is cut off there.
    """
    domains = [[x] if x else list(range(1, 10)) for x in flat]
    val = [0] * 81
    steps = [0]

    def dfs(k):
        steps[0] += 1
        if steps[0] >= cap:
            raise _StepCapReached
        if k == 81:
            return True
        for x in list(domains[k]):
            if not all(val[p] != x for p in _SUDOKU_PEERS[k]):
                continue
            val[k] = x
            pruned, wipeout = [], False
            for p in _SUDOKU_PEERS[k]:
                if not val[p] and x in domains[p]:
                    if len(domains[p]) == 1:
                        wipeout = True
                        break
                    pruned.append(p)
            if not wipeout:
                for p in pruned:
                    domains[p].remove(x)
                if dfs(k + 1):
                    return True
                for p in pruned:
                    domains[p].append(x)
            val[k] = 0
        return False

    try:
        dfs(0)
    except _StepCapReached:
        return cap
    return steps[0]


def _count_sudoku_solutions(flat, limit=2):
    """Count solutions up to `limit`, using bitmasks and most-constrained-cell-first."""
    val = list(flat)
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, x in enumerate(val):
        if x:
            b = 1 << x
            r, c = divmod(i, 9)
            if (rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]) & b:
                return 0  # two clues conflict
            rows[r] |= b
            cols[c] |= b
            boxes[(r // 3) * 3 + c // 3] |= b
    count = [0]

    def rec():
        best, best_mask, best_size = None, 0, 10
        for i in range(81):
            if not val[i]:
                r, c = divmod(i, 9)
                mask = ~(rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]) & 0x3FE
                size = bin(mask).count("1")
                if size == 0:
                    return
                if size < best_size:
                    best, best_mask, best_size = i, mask, size
        if best is None:
            count[0] += 1
            return
        r, c = divmod(best, 9)
        bx = (r // 3) * 3 + c // 3
        while best_mask:
            b = best_mask & -best_mask
            best_mask ^= b
            val[best] = b.bit_length() - 1
            rows[r] |= b
            cols[c] |= b
            boxes[bx] |= b
            rec()
            val[best] = 0
            rows[r] ^= b
            cols[c] ^= b
            boxes[bx] ^= b
            if count[0] >= limit:
                return

    rec()
    return count[0]


def _check_sudoku_csp_structure(csp, flat):
    cells = [(r, c) for r in range(9) for c in range(9)]
    assert list(csp.variables) == cells, \
        "csp.variables should be the 81 (row, col) tuples in row-major order: (0, 0), (0, 1), ..., (8, 8)."
    for (r, c), x in zip(cells, flat):
        expected = [x] if x else list(range(1, 10))
        assert list(csp.domains[(r, c)]) == expected, \
            "The domain of cell %r should be %r, but it is %r." % ((r, c), expected, csp.domains[(r, c)])
    for i, cell in enumerate(cells):
        peers = {cells[j] for j in _SUDOKU_PEERS[i]}
        assert set(csp.neighbors[cell]) == peers, \
            ("Cell %r should be constrained with exactly the 20 cells sharing its row, column, "
             "or 3x3 box. Missing: %r. Unexpected: %r."
             % (cell, sorted(peers - set(csp.neighbors[cell])), sorted(set(csp.neighbors[cell]) - peers)))
    for i, u in enumerate(cells):
        for j in _SUDOKU_PEERS[i]:
            if j < i:
                continue
            v = cells[j]
            between = [con for con in csp.constraints[u] if v in con.scope]
            for x in range(1, 10):
                for y in range(1, 10):
                    ok = all(con.satisfied({u: x, v: y}) for con in between)
                    assert ok == (x != y), \
                        ("The constraints between %r and %r should allow different digits and forbid "
                         "equal ones, but %r=%d, %r=%d is %s."
                         % (u, v, u, x, v, y, "allowed" if ok else "forbidden"))


# Function for tests
def test_ok():
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Test passed!!</strong>
        </div>""", raw=True)
    except:
        print("test ok!!")

class TestPSet4(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(15)
    def test_01_naive_search(self):
        australia_map_coloring, australia_map_coloring_impossible, naive_search, is_complete_and_valid = get_locals(self.notebook_locals, ["australia_map_coloring", "australia_map_coloring_impossible", "naive_search", "is_complete_and_valid"])

        csp = australia_map_coloring()
        solution, steps = naive_search(csp)

        assert solution is not None, "No solution returned for feasible Australia map"
        assert is_complete_and_valid(csp, solution), "Returned assignment is not a complete, valid coloring"
        assert steps > 10, "Naive Search should explore many many nodes - make sure you did not implement backtracking or forward checking instead"

        csp_impossible = australia_map_coloring_impossible()
        solution_impossible, steps_impossible = naive_search(csp_impossible)

        assert solution_impossible is None, "Solution incorrectly returned for impossible Australia map"
        assert steps_impossible == 255, "Naive Search should explore EVERY node before determining that there is no solution"

        test_ok()

    @weight(10)
    def test_02_backtracking_search(self):
        australia_map_coloring, australia_map_coloring_impossible, naive_search, backtracking_search, is_complete_and_valid = get_locals(self.notebook_locals, ["australia_map_coloring", "australia_map_coloring_impossible", "naive_search", "backtracking_search", "is_complete_and_valid"])

        csp = australia_map_coloring()
        solution_naive, steps_naive = naive_search(csp)
        solution_bt, steps_bt = backtracking_search(csp)

        assert solution_bt is not None, "No solution returned for feasible Australia map"
        assert is_complete_and_valid(csp, solution_bt), "Returned assignment is not a complete, valid coloring"
        assert steps_bt < steps_naive, "Backtracking should explore fewer nodes than naive search"

        csp_impossible = australia_map_coloring_impossible()
        solution_impossible_naive, steps_impossible_naive = naive_search(csp_impossible)
        solution_impossible_bt, steps_impossible_bt = backtracking_search(csp_impossible)

        assert solution_impossible_bt is None, "Solution incorrectly returned for impossible Australia map"
        assert steps_impossible_bt < steps_impossible_naive, "Backtracking should explore fewer nodes than naive search on the impossible Australia map"

        test_ok()

    @weight(20)
    def test_03_forward_checking(self):
        australia_map_coloring, australia_map_coloring_impossible, backtracking_search, forward_checking_search, is_complete_and_valid = get_locals(self.notebook_locals, ["australia_map_coloring", "australia_map_coloring_impossible", "backtracking_search", "forward_checking_search", "is_complete_and_valid"])

        csp = australia_map_coloring()
        solution_bt, steps_bt = backtracking_search(csp)
        solution_fc, steps_fc = forward_checking_search(csp)

        assert solution_fc is not None, "No solution returned for feasible Australia map"
        assert is_complete_and_valid(csp, solution_fc), "Returned assignment is not a complete, valid coloring"
        assert steps_fc <= steps_bt, "Forward checking should explore at most the same number of nodes as backtracking"

        csp_impossible = australia_map_coloring_impossible()
        solution_impossible_bt, steps_impossible_bt = backtracking_search(csp_impossible)
        solution_impossible_fc, steps_impossible_fc = forward_checking_search(csp_impossible)

        assert solution_impossible_fc is None, "Solution incorrectly returned for impossible Australia map"
        assert steps_impossible_fc < steps_impossible_bt, "For the impossible Australia map, forward checking should explore fewer nodes than backtracking"

    @weight(3)
    @timeout_decorator.timeout(60.0)
    def test_04_sudoku_csp(self):
        sudoku_csp, forward_checking_search = get_locals(
            self.notebook_locals, ["sudoku_csp", "forward_checking_search"])

        for name, puzzle in [("easy", SUDOKU_EASY), ("medium", SUDOKU_MEDIUM)]:
            flat = _flatten_sudoku(_as_grid(puzzle))
            csp = sudoku_csp(_as_grid(puzzle))
            _check_sudoku_csp_structure(csp, flat)

            solution, _ = forward_checking_search(csp)
            assert solution is not None, \
                "Forward checking found no solution for the %s test puzzle, which has one." % name
            solved = [solution.get((r, c)) for r in range(9) for c in range(9)]
            assert all(x == y for x, y in zip(flat, solved) if x), \
                "The solution to the %s test puzzle changes one of its clues." % name
            for i in range(81):
                for j in _SUDOKU_PEERS[i]:
                    assert solved[i] != solved[j], \
                        "The solution to the %s test puzzle repeats a digit in a row, column, or box." % name

        # The encoding should accept a numpy array as well as a list of lists.
        flat = _flatten_sudoku(_as_grid(SUDOKU_EASY))
        _check_sudoku_csp_structure(sudoku_csp(np.array(_as_grid(SUDOKU_EASY))), flat)

        solution, _ = forward_checking_search(sudoku_csp(_as_grid(SUDOKU_CONFLICTING)))
        assert solution is None, \
            "A puzzle with two 5s in the same row has no solution, but a solution was returned."

        test_ok()

    @weight(6)
    @timeout_decorator.timeout(300.0)
    def test_05_adversarial_sudoku(self):
        return_adversarial_sudoku = get_locals(self.notebook_locals, ["return_adversarial_sudoku"])
        flat = _flatten_sudoku(return_adversarial_sudoku())

        clues = sum(1 for x in flat if x)
        assert clues >= SUDOKU_MIN_CLUES, \
            "Your puzzle has %d clues; it needs at least %d." % (clues, SUDOKU_MIN_CLUES)

        n_solutions = _count_sudoku_solutions(flat, limit=2)
        assert n_solutions == 1, \
            ("Your puzzle has no solution." if n_solutions == 0 else
             "Your puzzle has more than one solution; it needs exactly one.")

        # Reference implementations of 1B and 1C, with the same row-major variable
        # order, ascending value order, and node counting.
        fc_steps = _reference_forward_checking_steps(flat, cap=SUDOKU_FC_MAX_STEPS + 1)
        assert fc_steps <= SUDOKU_FC_MAX_STEPS, \
            ("Forward checking visited more than %d nodes on your puzzle; it should visit at most %d."
             % (SUDOKU_FC_MAX_STEPS, SUDOKU_FC_MAX_STEPS))

        bt_steps = _reference_backtracking_steps(flat, cap=SUDOKU_BT_MIN_STEPS)
        assert bt_steps >= SUDOKU_BT_MIN_STEPS, \
            ("Backtracking solved your puzzle after visiting %d nodes; it should need at least %d."
             % (bt_steps, SUDOKU_BT_MIN_STEPS))

        print("Forward checking: %d nodes. Backtracking: at least %d nodes." % (fc_steps, bt_steps))
        test_ok()

    @weight(5)
    def test_08_planning_warmup(self):
        planning_warmup = get_locals(self.notebook_locals, ["planning_warmup"])
        plan = planning_warmup()
        assert len(plan) == 8
        assert plan[0].name == '(unstack a b)'
        test_ok()

    @weight(10)
    def test_09_pddl_warmup(self):
        pddl_warmup = get_locals(self.notebook_locals, ["pddl_warmup"])
        domain, problem = pddl_warmup()
        plan = run_planning(domain, problem, "gbf", "hadd")
        assert plan, "Failed to find a plan."
        picked_up_papers = set()
        satisfied_locs = set()
        for op in plan:
            if "pickup" in op.name:
                _, _, paper, _ = op.name.split(" ")
                assert paper not in picked_up_papers, \
                    "Should not pick up the same paper twice"
                picked_up_papers.add(paper)
            elif "deliver" in op.name:
                _, loc = op.name.rsplit(" ", 1)
                assert loc.endswith(")")
                loc = loc[:-1]
                assert loc not in satisfied_locs, \
                    "Should not deliver to the same place twice"
                satisfied_locs.add(loc)
        assert satisfied_locs == {"loc-1", "loc-2", "loc-3", "loc-4"}
        test_ok()
        
    @weight(10)
    def test_10(self):
        q10_answer = get_locals(self.notebook_locals, ["q10_answer"])
        answer =  (True, False, True, True, True, False, True, True, True)
        assert len(q10_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q10_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_11_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Eve".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
