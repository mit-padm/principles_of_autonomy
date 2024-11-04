import unittest
import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
# from nose.tools import assert_equal

from principles_of_autonomy.grader import get_locals
import numpy as np
import sympy
import random

import os
import time
import tempfile
from pyperplan.pddl.parser import Parser
from pyperplan import grounding, planner

# helper functions
# This uses TYPING
BLOCKS_DOMAIN = """(define (domain blocks)
    (:requirements :strips :typing)
    (:types block)
    (:predicates
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty)
        (holding ?x - block)
    )

    (:action pick-up
        :parameters (?x - block)
        :precondition (and
            (clear ?x)
            (ontable ?x)
            (handempty)
        )
        :effect (and
            (not (ontable ?x))
            (not (clear ?x))
            (not (handempty))
            (holding ?x)
        )
    )

    (:action put-down
        :parameters (?x - block)
        :precondition (and
            (holding ?x)
        )
        :effect (and
            (not (holding ?x))
            (clear ?x)
            (handempty)
            (ontable ?x))
        )

    (:action stack
        :parameters (?x - block ?y - block)
        :precondition (and
            (holding ?x)
            (clear ?y)
        )
        :effect (and
            (not (holding ?x))
            (not (clear ?y))
            (clear ?x)
            (handempty)
            (on ?x ?y)
        )
    )

    (:action unstack
        :parameters (?x - block ?y - block)
        :precondition (and
            (on ?x ?y)
            (clear ?x)
            (handempty)
        )
        :effect (and
            (holding ?x)
            (clear ?y)
            (not (clear ?x))
            (not (handempty))
            (not (on ?x ?y))
        )
    )
)
"""

# This uses TYPING
BLOCKS_PROBLEM = """(define (problem blocks)
    (:domain blocks)
    (:objects
        d - block
        b - block
        a - block
        c - block
    )
    (:init
        (clear a)
        (on a b)
        (on b c)
        (on c d)
        (ontable d)
        (handempty)
    )
    (:goal (and (on d c) (on c b) (on b a)))
)
"""

# The BW domain does not use TYPING
BW_BLOCKS_DOMAIN = """(define (domain prodigy-bw)
  (:requirements :strips)
  (:predicates (on ?x ?y)
               (ontable ?x)
               (clear ?x)
               (handempty)
               (holding ?x)
               )
  (:action pick-up
             :parameters (?ob1)
             :precondition (and (clear ?ob1) (ontable ?ob1) (handempty))
             :effect
             (and (not (ontable ?ob1))
                   (not (clear ?ob1))
                   (not (handempty))
                   (holding ?ob1)))
  (:action put-down
             :parameters (?ob)
             :precondition (holding ?ob)
             :effect
             (and (not (holding ?ob))
                   (clear ?ob)
                   (handempty)
                   (ontable ?ob)))
  (:action stack
             :parameters (?sob ?sunderob)
             :precondition (and (holding ?sob) (clear ?sunderob))
             :effect
             (and (not (holding ?sob))
                   (not (clear ?sunderob))
                   (clear ?sob)
                   (handempty)
                   (on ?sob ?sunderob)))
  (:action unstack
             :parameters (?sob ?sunderob)
             :precondition (and (on ?sob ?sunderob) (clear ?sob) (handempty))
             :effect
             (and (holding ?sob)
                   (clear ?sunderob)
                   (not (clear ?sob))
                   (not (handempty))
                   (not (on ?sob ?sunderob)))))
"""


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

class TestPSet8(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    def test_1(self):
        q1_answer = get_locals(self.notebook_locals, ["q1_answer"])
        answer = 1
        assert q1_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_2(self):
        q2_answer = get_locals(self.notebook_locals, ["q2_answer"])
        answer = (True, False, True, True, True, False, True, False)
        assert q2_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_3(self):
        q3_answer = get_locals(self.notebook_locals, ["q3_answer"])
        answer = (True, False, True, True, True, True, True, True)
        assert q3_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_4(self):
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        answer = (False, True, False)
        assert q4_answer == answer, "Incorrect values."

        test_ok()

    @weight(10)
    def test_5(self):
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        answer =  ('F', 'V', 'T', 'F', 'F', 'T', 'U')
        assert q5_answer == answer, "Incorrect values."

        test_ok()


    @weight(5)
    def test_6(self):
        warmup = get_locals(self.notebook_locals, 
                                                  ["warmup"])
        assert warmup() == [[4, -5, -6], [6, 5, -1], [2, 3]]

        test_ok()

    @weight(30)
    def test_7(self):
        run_inference_dpll = get_locals(self.notebook_locals,
                                           ["run_inference_dpll"])
        assert run_inference_dpll([[-1, 2]]) in [(True, {1: True, 2: True}), (True, {1: False, 2: True}), (True, {1: False, 2: False})]
        assert run_inference_dpll([[1], [-1]]) == (False, None)
        assert run_inference_dpll([[1, 2, 3], [-1, -2, -3], [1, -2, 3], [-1], [-3]]) == (False, None)
        assert run_inference_dpll([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32]]) == (True, {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True, 16: True, 17: True, 18: True, 19: True, 20: True, 21: True, 22: True, 23: True, 24: True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True})


        test_ok()

    @weight(5)
    def test_8(self):
        formula1_is_satisfiable = get_locals(self.notebook_locals, ["formula1_is_satisfiable"])
        assert formula1_is_satisfiable() == True

        test_ok()

    @weight(5)
    def test_9(self):
        formula2_is_satisfiable = get_locals(self.notebook_locals, ["formula2_is_satisfiable"])
        assert formula2_is_satisfiable() == False

        test_ok()

    @weight(20)
    def test_10(self):
        infer_unknown_values = get_locals(self.notebook_locals, ["infer_unknown_values"])
        assert infer_unknown_values([["U", "C", "C"], ["S", "C", "U"], ["U", "U", "C"]]) == [["C", "C", "C"], ["S", "C", "C"], ["F", "S", "C"]]
        assert infer_unknown_values([["U", "S", "C", "U"], ["U", "U", "C", "U"], ["U", "S", "C", "U"]]) == [["F", "S", "C", "C"], ["S", "C", "C", "C"], ["F", "S", "C", "C"]]
        assert infer_unknown_values([["U", "U", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "F", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"]]) == [["C", "C", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "S", "U", "U", "U", "U"], ["U", "U", "S", "F", "S", "U", "U", "U"], ["U", "U", "U", "S", "U", "U", "U", "U"]]

        test_ok()

                
    @weight(5)
    def test_11(self):
        q11_answer = get_locals(self.notebook_locals, ["q11_answer"])
        answer =  (True, True, False)
        assert q11_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_12(self):
        q12_answer = get_locals(self.notebook_locals, ["q12_answer"])
        answer =  False
        assert q12_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_13(self):
        q13_answer = get_locals(self.notebook_locals, ["q13_answer"])
        answer =  False
        assert q13_answer == answer, "Incorrect values."

        test_ok()
    
    @weight(5)
    def test_14(self):
        q14_answer = get_locals(self.notebook_locals, ["q14_answer"])
        answer =  True
        assert q14_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_15(self):
        planning_warmup = get_locals(self.notebook_locals, ["planning_warmup"])
        plan = planning_warmup()
        assert len(plan) == 8
        assert plan[0].name == '(unstack a b)'
        test_ok()

    @weight(10)
    def test_16(self):
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
    def test_17(self):
        q17_answer = get_locals(self.notebook_locals, ["q17_answer"])
        answer =  (True, False, True, True, True, False, True, True, True)
        assert q17_answer == answer, "Incorrect values."

        test_ok()
    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_18_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Hawaii".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
