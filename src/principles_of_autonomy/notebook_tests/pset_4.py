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
        assert steps_impossible > 10, "Naive Search should explore many many nodes - make sure you did not implement backtracking or forward checking instead"

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

    @weight(5)
    def test_04(self):
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        answer = (True, True, False)
        assert len(q4_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q4_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_05(self):
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        answer = False
        assert q5_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_06(self):
        q6_answer = get_locals(self.notebook_locals, ["q6_answer"])
        answer = False
        assert q6_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_07(self):
        q7_answer = get_locals(self.notebook_locals, ["q7_answer"])
        answer = True
        assert q7_answer == answer, "Incorrect values."

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
        password_hash = hash("Dunkin Donuts".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
