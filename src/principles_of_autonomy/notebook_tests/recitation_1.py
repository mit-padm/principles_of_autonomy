import unittest
import numpy as np 
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight

from principles_of_autonomy.grader import get_locals

class TestRecitation1(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    
    @weight(5)
    @timeout_decorator.timeout(5.0)
    def test_1_check_square_function(self):
        square_values = get_locals(self.notebook_locals, ["square_values"])
        values = [1,2,4,8]
        out = square_values(values)
        for in_val, out_val in zip(values, out):
            assert np.isclose(in_val**2, out_val)

    @weight(5)
    @timeout_decorator.timeout(5.0)
    def test_2_check_fast_square_function(self):
        fast_square_values = get_locals(self.notebook_locals, ["fast_square_values"])
        values = np.arange(0,100)
        out = fast_square_values(values)
        assert np.allclose(values**2, out)

    @weight(5)
    @timeout_decorator.timeout(5.0)
    def test_3_robot_class(self):
        Robot = get_locals(self.notebook_locals, ["Robot"])
        robot = Robot("test_robot", False)
        robot.move_left()
        assert robot.get_position() == 1
        robot.move_right()
        robot.move_right()
        assert robot.get_position() == -1

        robot = Robot("test_robot", True)
        robot.move_left()
        assert robot.get_position() == -1
        robot.move_right()
        robot.move_right()
        assert robot.get_position() == 1

        robot.move_right()
        robot.move_right()
        robot.move_right()
        robot.move_right()
        robot.move_right()
        robot.move_right()
        robot.reset()
        assert robot.get_position() == 0
