import unittest
import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
# from nose.tools import assert_equal

from principles_of_autonomy.grader import get_locals
import random
import numpy as np
import random

# Function for tests
def test_ok():
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Test passed!!</strong>
        </div>""", raw=True)
    except:
        print("test ok!!")

class TestPSet6(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    def test_1(self):
        q1_answer = get_locals(
            self.notebook_locals, ["q1_answer"])
        # Assert that the computed values match the expected values
        assert q1_answer == (1, 2, 4, 5, 7), "Incorrect values."

        test_ok()

    @weight(5)
    def test_2(self):
        q2_answer = get_locals(
            self.notebook_locals, ["q2_answer"])
        # Assert that the computed values match the expected values
        assert q2_answer == (1, 3, 6), "Incorrect values."

        test_ok()

    @weight(5)
    def test_3(self):
        q3_answer = get_locals(
            self.notebook_locals, ["q3_answer"])
        # Assert that the computed values match the expected values
        assert abs(q3_answer - .8) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_4(self):
        q4_answer = get_locals(
            self.notebook_locals, ["q4_answer"])
        # Assert that the computed values match the expected values
        assert abs(q4_answer - .875) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_5(self):
        q5_answer = get_locals(
            self.notebook_locals, ["q5_answer"])
        # Assert that the computed values match the expected values
        assert abs(q5_answer - .76) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_6(self):
        q6_answer = get_locals(
            self.notebook_locals, ["q6_answer"])
        # Assert that the computed values match the expected values
        assert abs(q6_answer - .613) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_7(self):
        q7_answer = get_locals(
            self.notebook_locals, ["q7_answer"])
        # Assert that the computed values match the expected values
        assert q7_answer == 8, "Incorrect values."

        test_ok()

    @weight(5)
    def test_8(self):
        q8_answer = get_locals(
            self.notebook_locals, ["q8_answer"])
        # Assert that the computed values match the expected values
        assert abs(q8_answer - .0005) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_9(self):
        q9_answer = get_locals(
            self.notebook_locals, ["q9_answer"])
        # Assert that the computed values match the expected values
        assert abs(q9_answer - .07781) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_10(self):
        q10_answer = get_locals(
            self.notebook_locals, ["q10_answer"])
        # Assert that the computed values match the expected values
        assert q10_answer == (0, 0.1, 0, 1), "Incorrect values."

        test_ok()

    @weight(5)
    def test_11(self):
        q11_answer = get_locals(
            self.notebook_locals, ["q11_answer"])
        # Assert that the computed values match the expected values
        assert q11_answer == (0.1, 1), "Incorrect values."

        test_ok()

    @weight(5)
    def test_12(self):
        q12_answer = get_locals(
            self.notebook_locals, ["q12_answer"])
        # Assert that the computed values match the expected values
        assert q12_answer == 4, "Incorrect values."

        test_ok()

    @weight(5)
    def test_13(self):
        q13_answer = get_locals(
            self.notebook_locals, ["q13_answer"])
        # Assert that the computed values match the expected values
        assert q13_answer == 3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_14_BN_warmup(self):
        BN_warmup = get_locals(
            self.notebook_locals, ["BN_warmup"])
        my_CPT = BN_warmup()
        assert isinstance(my_CPT.rvs, (tuple, list))
        assert len(my_CPT.rvs) == 2
        rain_rv, cloud_rv = None, None
        for rv in my_CPT.rvs:
            assert rv.dim == 2
            if rv.name == 'Rain':
                rain_rv = rv
            elif rv.name == 'Clouds':
                cloud_rv = rv
            else:
                assert False, "Unexpected RV name"
        assert abs(my_CPT.get_by_names({'Rain': 0, 'Clouds': 0}) - 0.8) < 1e-3
        assert abs(my_CPT.get_by_names({'Rain': 1, 'Clouds': 0}) - 0.2) < 1e-3
        assert abs(my_CPT.get_by_names({'Rain': 0, 'Clouds': 1}) - 0.5) < 1e-3
        assert abs(my_CPT.get_by_names({'Rain': 1, 'Clouds': 1}) - 0.5) < 1e-3

        test_ok()

    @weight(5)
    def test_15_BN_warmup2(self):
        BN_warmup2, RV, CPT = get_locals(
            self.notebook_locals, ["BN_warmup2", "RV", "CPT"])

        A = RV("A", [0, 1])
        B = RV("B", [0, 1])
        warmup2_CPT = CPT(
            rvs=[A, B],
            table=np.array([
                [0.98, 0.01],
                [0.02, 0.99],
            ])
        )
        assert abs(BN_warmup2(warmup2_CPT) - 0.01) < 1e-3

        test_ok()


    @weight(5)
    def test_16_BN_multiply(self):
        multiply_tables, RV, CPT = get_locals(
            self.notebook_locals, ["multiply_tables", "RV", "CPT"])

        A, B = RV('A', [0, 1, 2]), RV('B', [0, 1])
        # A matches B
        cpt1 = CPT([A, B],
                   np.array([
                       [1, 0],
                       [0, 1],
                       [0, 0],
                   ]))
        # B is 0
        cpt2 = CPT([B], np.array([1, 0]))
        # A and B are both 0
        expected_result_table = np.zeros((3, 2))
        expected_result_table[0, 0] = 1
        expected_result = CPT([A, B], expected_result_table)
        result = multiply_tables([cpt1, cpt2])
        assert result.allclose(expected_result)

        test_ok()


    @weight(5)
    def test_17_BN_marginalize(self):
        marginalize, RV, CPT = get_locals(
            self.notebook_locals, ["marginalize", "RV", "CPT"])
        A, B, C = RV('A', [0, 1]), RV('B', [0, 1, 2]), RV('C', [5, 4, 3, 2])
        CPT_table = np.array([
        [[0.74656105, 0.48018556, 0.84608715, 0.91133775],
         [0.61895184, 0.57285163, 0.01740987, 0.99736661],
         [0.93160191, 0.55531759, 0.51517323, 0.78838295]],
        [[0.42734642, 0.91677512, 0.334311, 0.06248867],
         [0.80557158, 0.48754704, 0.00403897, 0.36942851],
         [0.81986122, 0.75209997, 0.77740444, 0.6694172]]])
        cpt = CPT([A, B, C], CPT_table)
        expected_result = CPT([A],
                              np.array([7.98122714, 6.42629014]))
        result = marginalize(cpt, {C, B})
        assert result.allclose(expected_result)

        test_ok()


    @weight(10)
    def test_18_BN_inference(self):
        BN_warmup, RV, CPT, is_it_cloudy = get_locals(
            self.notebook_locals, ["BN_warmup", "RV", "CPT", "is_it_cloudy"])
        rain_model = BN_warmup()
        e = CPT(rvs=[RV("Rain", [0, 1])], table=np.array([1, 0]))
        result = is_it_cloudy(rain_model, e)
        assert abs(result.get_by_names({'Clouds': 0}) - 0.6153) < 1e-3
        assert abs(result.get_by_names({'Clouds': 1}) - 0.3846) < 1e-3

        test_ok()

        
    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_19_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Finland".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
