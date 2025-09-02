import unittest
import numpy as np 
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight

from principles_of_autonomy.grader import get_locals

def check_expanded_states(returned_states, correct_states):
    assert isinstance(returned_states, list), "Your function should return a list."
    assert len(returned_states)==len(correct_states),"Your function returned %d states, but it should have returned %d"%(len(returned_states),len(correct_states))
    for s in returned_states:
        assert isinstance(s, tuple),"Each state returned by your function should be a tuple. %s is not" %(s,)
        assert len(s)==3, "Each state should have three internal tuples, %s doesn't." %(s,)
        for l in s:
            assert isinstance(l, tuple),"Each state should have three internal tuples, %s doesn't." %(s,)
            assert len(l)==3, "Each internal tuple of a state should have three elements, %s doesn't" %(s,)

    for correct_state in correct_states:
        assert correct_state in returned_states, "%s state is not in returned states, and it should be."%(correct_state,)
    


class TestPSet1(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    
    @weight(25)
    @timeout_decorator.timeout(5.0)
    def test_1_check_expanded_states(self):
        expand_state = get_locals(self.notebook_locals, ["expand_state"])
        check_expanded_states(expand_state(((0, 1, 3), (4, 2, 5), (7, 8, 6))),
                            [((4, 1, 3), (0, 2, 5), (7, 8, 6)), ((1, 0, 3), (4, 2, 5), (7, 8, 6))])

        check_expanded_states(expand_state(((1, 2, 3), (8, 0, 4), (7, 6, 5))),
                            [((1, 2, 3), (8, 6, 4), (7, 0, 5)),
                            ((1, 0, 3), (8, 2, 4), (7, 6, 5)),
                            ((1, 2, 3), (8, 4, 0), (7, 6, 5)),
                            ((1, 2, 3), (0, 8, 4), (7, 6, 5))])

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_2_puzzle_problem_expanded_nodes(self):
        PuzzleProblem, SearchNode = get_locals(self.notebook_locals, ["PuzzleProblem", "SearchNode"])
        state_test = ((1, 2, 3), (8, 0, 4), (7, 6, 5))
        problem_test = PuzzleProblem(state_test, None)
        node_test = SearchNode(state_test, None)


        def check_expanded_nodes(returned_nodes, parent_node, correct_states):
            assert isinstance(returned_nodes, list), "Your function should return a list."
            assert len(returned_nodes)==len(correct_states),"Your function returned %d nodes, but it should have returned %d"%(len(returned_nodes),
                        len(correct_states))

            for n in returned_nodes:
                assert isinstance(n, SearchNode), "Your function should return a list of SearchNodes."
                assert n.parent is parent_node, "The parent node for node %s is wrong." % n

            check_expanded_states([n.state for n in returned_nodes], correct_states)

        check_expanded_nodes(problem_test.expand_node(node_test),
                            node_test,
                            [((1, 2, 3), (8, 6, 4), (7, 0, 5)),
                            ((1, 0, 3), (8, 2, 4), (7, 6, 5)),
                            ((1, 2, 3), (8, 4, 0), (7, 6, 5)),
                            ((1, 2, 3), (0, 8, 4), (7, 6, 5))])
        
    @weight(35)
    @timeout_decorator.timeout(5.0)
    def test_3_bfs(self):
        PuzzleProblem, breadth_first_search, print_state = get_locals(self.notebook_locals,["PuzzleProblem", "breadth_first_search", "print_state"])
        start_state = ((0, 1, 3), (4, 2, 5), (7, 8, 6))
        goal_state = ((1,2,3),(4,5,6),(7,8,0))
        problem = PuzzleProblem(start_state, goal_state)

        sol, num_visited, max_q = breadth_first_search(problem)
        if sol:
            assert len(sol.path) == 5, "The shortest path involves 5 moves"
            assert sol.path[0] == start_state, "The first state in the path is not the start state"
            assert sol.path[-1] == goal_state, "The last state in the path is not the goal state"
            print("Solution: ")
            for s in sol.path:
                print_state(s)
                print("\n**\n")
        else:
            print("No solution after exploring %d states with max q of %d" %(num_visited, max_q))

    @weight(25)
    @timeout_decorator.timeout(5.0)
    def test_4_dfs(self):
        SearchNode, depth_first_search, print_state = get_locals(self.notebook_locals,["SearchNode", "depth_first_search", "print_state"])

        def inst_expand_state(state, expand_list):
            def swap(ori_idx, dest_idx):
                list_state = list([list(l) for l in state])
                list_state[ori_idx[0]][ori_idx[1]], list_state[dest_idx[0]][dest_idx[1]] = \
                    list_state[dest_idx[0]][dest_idx[1]], list_state[ori_idx[0]][ori_idx[1]]
                return tuple(tuple(l) for l in list_state)
                        
            hole_idx = (-1,-1)
            for (i,l) in enumerate(state):
                for (j,el) in enumerate(l):
                    if el==0:
                        hole_idx = (i,j)
                        break
            dest_idx = [(hole_idx[0]+a, hole_idx[1]+b) for a,b in expand_list]
            valid_idx = lambda idx: 0<=idx[0]<=2 and 0<=idx[1]<=2
            dest_idx = filter(valid_idx, dest_idx)
            states = [swap(hole_idx, d_idx) for d_idx in dest_idx]
            return states

        class InstPuzzleProblem(object):
            """Class that represents the puzzle search problem."""
            def __init__(self, start, goal, expand_list):
                self.start = start
                self.goal = goal
                self.expand_list = expand_list
            def test_goal(self, state):
                return self.goal == state
            def expand_node(self, search_node):
                """Return a list of SearchNodes, having the correct state and parent node."""
                expanded_sn = [SearchNode(new_state, search_node)
                            for new_state in inst_expand_state(search_node.state, self.expand_list)]
                return expanded_sn

        minimal_start = ((1,2,3),(4,5,6),(7,0,8))
        minimal_goal = ((1,2,3),(4,5,6),(7,8,0))

        right_last_in = [(-1,0),(0,-1),(1,0),(0,1)]
        problem = InstPuzzleProblem(minimal_start, minimal_goal, right_last_in)
        sol, num_visited, max_q = depth_first_search(problem)

        right_first_in = [(0,1),(1,0),(0,-1),(-1,0)]
        problem = InstPuzzleProblem(minimal_start, minimal_goal, right_first_in)
        backtrack_sol, backtrack_num_visited, backtrack_max_q = depth_first_search(problem)

        if sol:
            assert sol and len(sol.path) == 2, "Expanding nodes in an ideal order did not return the expected 2-state solution"
            assert sol.path[0] == minimal_start, "The first state in the path is not the start state"
            assert sol.path[-1] == minimal_goal, "The last state in the path is not the goal state"
            assert len(backtrack_sol.path) > len(sol.path), "Expanding nodes in a non-ideal order returned the shortest solution. Make sure you wrote DFS, not BFS!"
            assert backtrack_num_visited > len(backtrack_sol.path), "DFS did not backtrack"
            print("Solution: ")
            for s in sol.path:
                print_state(s)
                print("\n**\n")
        else:
            print("No solution after exploring %d states with max q of %d" %(num_visited, max_q))

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_5_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Apple Pie".lower())
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")