import unittest
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
        print("test ok!!")

class TestPSet3(unittest.TestCase):
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
        assert len(q2_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q2_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_3(self):
        q3_answer = get_locals(self.notebook_locals, ["q3_answer"])
        answer = (True, False, True, True, True, True, True, True)
        assert len(q3_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q3_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_4(self):
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        answer = (False, True, False)
        assert len(q4_answer) == len(answer), f"Incorrect number of values, need {len(answer)} True / False values"
        assert q4_answer == answer, "Incorrect values."

        test_ok()

    @weight(10)
    def test_5(self):
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        answer =  ('F', 'V', 'T', 'F', 'F', 'T', 'U')
        assert len(q5_answer) == len(answer), f"Incorrect number of values, need {len(answer)} characters"
        assert q5_answer == answer, "Incorrect values."

        test_ok()


    @weight(5)
    def test_6(self):
        warmup = get_locals(self.notebook_locals, 
                                                  ["warmup"])
        answer = warmup()
        assert len(answer) == 3, "Incorrect number of clauses"
        assert len(answer[0])==3 and len(answer[1])==3 and len(answer[2])==2, "Incorrect number of literals in each clause"
        assert (4 in answer[0] and -5 in answer[0] and -6 in answer[0] and 
                6 in answer[1] and 5 in answer[1] and -1 in answer[1] and
                2 in answer[2] and 3 in answer[3]), "Incorrect literals"
        assert answer == [[4, -5, -6], [6, 5, -1], [2, 3]], "Incorrect order of literals"

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
        is_satisfiable, formula = formula1_is_satisfiable()
        assert is_satisfiable == True, "Incorrect boolean value"
        assert isinstance(formula, sympy.Expr), "Not a sympy formula"

        test_ok()

    @weight(5)
    def test_9(self):
        formula2_is_satisfiable = get_locals(self.notebook_locals, ["formula2_is_satisfiable"])
        is_satisfiable, formula = formula2_is_satisfiable()
        assert is_satisfiable == False, "Incorrect boolean value"
        assert isinstance(formula, sympy.Expr), "Not a sympy formula"

        test_ok()

    @weight(25)
    def test_10(self):
        infer_unknown_values = get_locals(self.notebook_locals, ["infer_unknown_values"])
        assert infer_unknown_values([["U", "C", "C"], ["S", "C", "U"], ["U", "U", "C"]]) == [["C", "C", "C"], ["S", "C", "C"], ["F", "S", "C"]]
        assert infer_unknown_values([["U", "S", "C", "U"], ["U", "U", "C", "U"], ["U", "S", "C", "U"]]) == [["F", "S", "C", "C"], ["S", "C", "C", "C"], ["F", "S", "C", "C"]]
        assert infer_unknown_values([["U", "U", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "F", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"]]) == [["C", "C", "C", "U", "U", "U", "U", "U"], ["C", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "U", "U", "U", "U", "U", "C", "C"], ["U", "C", "U", "S", "U", "U", "U", "U"], ["U", "U", "S", "F", "S", "U", "U", "U"], ["U", "U", "U", "S", "U", "U", "U", "U"]]

        test_ok()

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_11_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Ciulama".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
