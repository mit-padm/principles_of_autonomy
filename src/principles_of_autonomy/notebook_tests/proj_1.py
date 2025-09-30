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

class TestProj1(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    def test_01_warmup_1(self):
        State, sar_warmup1 = get_locals(self.notebook_locals, ["State", "sar_warmup1"])

        minimal_state_map = np.array([['C', 'F'], ['S', 'W']])
        sar_state = State(
            hospital = (1, 1),
            people = {"p1": (1, 0)},
            state_map = minimal_state_map,
        )

        assert sar_warmup1(sar_state, 0, 0) == False, "Free spaces should NOT be considered obstacles"
        assert sar_warmup1(sar_state, 0, 1) == False, "Fire should NOT be considered an obstacle"
        assert sar_warmup1(sar_state, 1, 0) == False, "Smoke should NOT be considered an obstacle"
        assert sar_warmup1(sar_state, 1, 1) == True, "Returned 'False' for a cell with an obstacle"

        test_ok()

    @weight(5)
    def test_02_warmup_2(self):
        sar_warmup2, SearchAndRescueProblem, execute_plan, State = get_locals(self.notebook_locals, ["sar_warmup2", "SearchAndRescueProblem", "execute_plan", "State"])
        
        problem = SearchAndRescueProblem()
        plan = sar_warmup2()
        state = execute_plan(problem, plan, State())
        assert state.people["p1"] == (6, 6), f"p1 not delivered to the hospital location, still at {state.people["p1"]}"

        test_ok()

    @weight(5)
    def test_03_sar_pddl(self):
        SearchAndRescuePlanner, SearchAndRescueProblem, State, execute_count_num_delivered = get_locals(self.notebook_locals, ["SearchAndRescuePlanner", "SearchAndRescueProblem", "State", "execute_count_num_delivered"])

        problem = SearchAndRescueProblem()
        planner = SearchAndRescuePlanner(search_algo="gbf", heuristic="hff")
        state = State()
        plan, plan_time = planner.get_plan(state)
        assert plan is not None, "Plan not found for feasible problem"
        assert execute_count_num_delivered(problem=problem, state=state, plan=plan, visualize=False) == 4, "All people not delivered to hospital"

        test_ok()

    @weight(10)
    def test_04_infer_unknown(self):
        infer_unknown_values = get_locals(self.notebook_locals, ["infer_unknown_values"])
        assert infer_unknown_values([["U", "F"]]) == [["U", "F"]]
        assert infer_unknown_values([["F", "U", "C"], ["S", "C", "U"], ["U", "U", "C"]]) == [["F", "S", "C"], ["S", "C", "C"], ["U", "U", "C"]]
        assert infer_unknown_values([["U", "C", "C"], ["S", "C", "U"], ["U", "U", "C"]]) == [["C", "C", "C"], ["S", "C", "C"], ["F", "S", "C"]]
        assert infer_unknown_values([["U", "S", "C", "U"], ["U", "U", "C", "U"], ["U", "S", "C", "U"]]) == [["F", "S", "C", "C"], ["U", "U", "C", "C"], ["F", "S", "C", "C"]]
        assert infer_unknown_values([["U", "U", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "F", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"]]) == [["C", "C", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "F", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"]]
        assert infer_unknown_values([["C", "U", "C", "U", "U", "C", "U"], ["U", "W", "W", "U", "C", "W", "W"], ["U", "F", "U", "U", "U", "F", "U"], ["C", "S", "W", "C", "U", "U", "U"], ["U", "U", "W", "U", "W", "U", "U"], ["C", "C", "U", "C", "U", "W", "U"], ["U", "W", "C", "U", "W", "U", "C"]]) == [["C", "C", "C", "C", "C", "C", "C"], ["C", "W", "W", "C", "C", "W", "W"], ["S", "F", "U", "U", "S", "F", "U"], ["C", "S", "W", "C", "U", "U", "U"], ["C", "C", "W", "C", "W", "U", "U"], ["C", "C", "C", "C", "C", "W", "U"], ["C", "W", "C", "C", "W", "C", "C"]]
        assert infer_unknown_values([["C", "U", "C", "U", "U", "C", "U"], ["U", "W", "W", "U", "C", "W", "W"], ["U", "F", "U", "U", "U", "F", "U"], ["C", "S", "W", "C", "U", "F", "U"], ["U", "U", "W", "U", "W", "U", "U"], ["C", "C", "U", "C", "U", "W", "F"], ["U", "W", "C", "U", "W", "U", "U"]]) == [["C", "C", "C", "C", "C", "C", "C"], ["C", "W", "W", "C", "C", "W", "W"], ["S", "F", "U", "U", "S", "F", "U"], ["C", "S", "W", "C", "S", "F", "U"], ["C", "C", "W", "C", "W", "U", "U"], ["C", "C", "C", "C", "C", "W", "F"], ["C", "W", "C", "C", "W", "U", "U"]]

        test_ok()

    @weight(10)
    def test_05_belief_update(self):
        SearchAndRescueProblem, State, BeliefState = get_locals(self.notebook_locals, ["SearchAndRescueProblem", "State", "BeliefState"])

        state_map = np.array([["C", "S", "C", "C", "C"], ["S", "F", "S", "C", "C"],
                            ["S", "F", "S", "S", "S"], ["S", "F", "F", "F", "F"],
                            ["C", "S", "S", "S", "S"], ["C", "C", "C", "C", "C"]])
        beliefstate_map = np.array([["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"]])
        problem = SearchAndRescueProblem()
        state = State(state_map=state_map)
        bel = BeliefState(state_map=beliefstate_map)
        observation = problem.get_observation(state)
        new_bel = bel.update(problem, observation)
        assert new_bel.robot == (0, 0)
        assert new_bel.state_map.tolist() == [['C', 'S', 'U', 'U', 'U'],
                                            ['S', 'U', 'U', 'U', 'U'],
                                            ['U', 'U', 'U', 'U', 'U'],
                                            ['U', 'U', 'U', 'U', 'U'],
                                            ['U', 'U', 'U', 'U', 'U'],
                                            ['U', 'U', 'U', 'U', 'U']]

        state_map = np.array([["C", "S", "C", "C", "C"], ["S", "F", "S", "C", "C"],
                            ["S", "F", "S", "S", "S"], ["S", "F", "F", "F", "F"],
                            ["C", "S", "S", "S", "S"], ["C", "C", "C", "C", "C"]])
        beliefstate_map = np.array([["U", "U", "U", "U", "U"],
                                    ["S", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"],
                                    ["U", "U", "U", "U", "U"]])
        problem = SearchAndRescueProblem()
        state = State(state_map=state_map)
        bel = BeliefState(state_map=beliefstate_map)

        new_state, _ = problem.get_next_state(state, 'down')
        observation = problem.get_observation(new_state)
        new_bel = bel.update(problem, observation, 'down')
        assert new_bel.robot == (1, 0)
        assert new_bel.state_map.tolist() == [['C', 'S', 'U', 'U', 'U'],
                                            ['S', 'F', 'U', 'U', 'U'],
                                            ['S', 'U', 'U', 'U', 'U'],
                                            ['U', 'U', 'U', 'U', 'U'],
                                            ['U', 'U', 'U', 'U', 'U'],
                                            ['U', 'U', 'U', 'U', 'U']]

        test_ok()

    @weight(7)
    def test_06_safe_not_smart(self):
        SearchAndRescueProblem, State, BeliefState, make_greedy_policy, agent_loop = get_locals(self.notebook_locals, ["SearchAndRescueProblem", "State", "BeliefState", "make_greedy_policy", "agent_loop"])

        problem = SearchAndRescueProblem()
        policy = make_greedy_policy(problem)

        # Empty map
        state = State()
        state.state_map[:, :] = 'C'
        bel = BeliefState()
        bel.state_map[:, :] = 'C'

        s_or_f, final_state, final_bel = agent_loop(problem, state, policy, bel, visualize=False)
        assert s_or_f == '*Success*' and final_state.robot == final_state.hospital, "Robot should be able to make it to the hospital in a clear map"

        problem = SearchAndRescueProblem()

        # Use default map
        state = State()
        bel = BeliefState()

        policy = make_greedy_policy(problem)
        s_or_f, final_state, final_bel = agent_loop(problem, state, policy, bel, visualize=False)
        r, c = final_state.robot
        hr, hc = final_state.hospital
        distance = abs(hr - r) + abs(hc - c)
        assert distance < 12, "Despite being not-so-smart, the robot should at least get closer to the hospital than its start position"

        test_ok()

    @weight(7)
    def test_07_safe_smart(self):
        SearchAndRescueProblem, SearchAndRescuePlanner, State, BeliefState, make_planner_policy, agent_loop, get_num_delivered = get_locals(self.notebook_locals, ["SearchAndRescueProblem", "SearchAndRescuePlanner", "State", "BeliefState", "make_planner_policy", "agent_loop", "get_num_delivered"])

        problem = SearchAndRescueProblem()
        base_planner = SearchAndRescuePlanner(search_algo="gbf", heuristic="hff")

        def planner(state):
            plan, time = base_planner.get_plan(state)
            return plan

        policy = make_planner_policy(problem, planner)
        state = State()
        # Observable
        bel = BeliefState(state_map=state.state_map)
        s_or_f, final_state, final_bel = agent_loop(problem, state, policy, bel, visualize=False)
        assert get_num_delivered(final_state) == 4, "All people not delivered to hospital"

        test_ok()