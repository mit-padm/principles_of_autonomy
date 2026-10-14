import unittest
from gradescope_utils.autograder_utils.decorators import weight

from principles_of_autonomy.grader import get_locals

# Project 2 is graded manually. This autograder only checks that the submitted
# notebook runs from top to bottom without errors; it awards no points.

class TestProj2(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(0)
    def test_00_notebook_executes(self):
        # Reaching this test means every cell ran without raising an exception.
        get_locals(self.notebook_locals, [
            "run_astar_search",
            "run_astar_search_faster",
            "FireMDPDeterminizedAStarAgent",
            "MCTSAgent",
        ])
