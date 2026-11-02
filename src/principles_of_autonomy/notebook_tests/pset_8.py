import hashlib
import unittest
import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight

from principles_of_autonomy.grader import get_locals
import random
import numpy as np
import random

# Problem 4: the fewest-edge Bayes net for the joint in hidden_joint.npy.
# Only the joint's hash and the minimum edge count live here, not the answer.
_HIDDEN_JOINT_SHA256 = "7cab9bd34e733b1aee8040266faa014280116c36c13932e26fe6519cb81766e8"
_HIDDEN_JOINT_MIN_EDGES = 12
# A correct net leaves ~1e-16 of dependence unexplained; leaving out any needed edge leaves >= ~1e-3.
_HIDDEN_JOINT_TOL = 1e-8


def _check_edge_list(edges, names):
    """Validate a student's edge list and return it as a list of (parent, child) name pairs."""
    assert isinstance(edges, (list, tuple, set)), \
        f"q19_answer should be a list of (parent, child) tuples, not {type(edges).__name__}."
    checked = []
    for edge in edges:
        assert isinstance(edge, (list, tuple)) and len(edge) == 2, \
            f"Each edge should be a (parent, child) tuple of variable names, but got {edge!r}."
        parent, child = edge
        assert parent in names and child in names, \
            f"Edge {edge!r} uses a name that is not one of the variables {names}."
        assert parent != child, f"Edge {edge!r} is a self-loop."
        checked.append((parent, child))
    assert len(set(checked)) == len(checked), "Your edge list contains a duplicate edge."
    return checked


def _topological_order(names, edges):
    """Order the variables so every parent comes before its children (fails on a cycle)."""
    parents = {x: {p for p, c in edges if c == x} for x in names}
    order = []
    while len(order) < len(names):
        ready = [x for x in names if x not in order and parents[x] <= set(order)]
        assert ready, ("Your edges contain a cycle, so they are not a Bayes net. "
                       f"These variables are on or after a cycle: {[x for x in names if x not in order]}.")
        order.append(ready[0])
    return order


def _missing_dependence(table, names, edges, order):
    """For each X, I(X; earlier variables | parents(X)) under `order`.

    These terms sum to the KL divergence from the joint to the product of P(X | parents(X)),
    so they are all ~0 exactly when the net represents the joint.
    """
    def entropy(variables):
        if not variables:
            return 0.0
        axes = tuple(i for i, x in enumerate(names) if x not in variables)
        p = table.sum(axis=axes).ravel()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    terms = {}
    for i, x in enumerate(order):
        parents = {p for p, c in edges if c == x}
        earlier = set(order[:i])
        terms[x] = (entropy(parents | {x}) - entropy(parents)
                    - entropy(earlier | {x}) + entropy(earlier))
    return terms

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
    def test_01(self):
        q1_answer = get_locals(self.notebook_locals, ["q1_answer"])
        answer = set((1, 2, 4, 5, 7))
        q1_set = set(q1_answer)

        assert q1_set <= answer, f"Selected too many values."
        assert q1_set >= answer, f"Did not select enough values."
        assert len(q1_set) == len(answer), f"Incorrect number of values."
        assert q1_set == answer, f"Incorrect values."

        test_ok()

    @weight(5)
    def test_02(self):
        q2_answer = get_locals(self.notebook_locals, ["q2_answer"])
        answer = set((1, 3, 6))
        q2_set = set(q2_answer)

        assert q2_set <= answer, f"Selected too many values."
        assert q2_set >= answer, f"Did not select enough values."
        assert len(q2_set) == len(answer), f"Incorrect number of values."
        assert q2_set == answer, f"Incorrect values."

        test_ok()

    @weight(5)
    def test_03(self):
        q3_answer = get_locals(self.notebook_locals, ["q3_answer"])
        answer = 0.8
        assert q3_answer >= 0 and q3_answer <= 1, "Probabilities should be between 0 and 1, inclusive."
        assert isinstance(q3_answer, float), "Probability should be a float accurate to 3 digits after the decimal point."
        assert abs(q3_answer - answer) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_04(self):
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        answer = 0.875
        assert q4_answer >= 0 and q4_answer <= 1, "Probabilities should be between 0 and 1, inclusive."
        assert isinstance(q4_answer, float), "Probability should be a float accurate to 3 digits after the decimal point."
        assert abs(q4_answer - answer) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_05(self):
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        answer = 0.76
        assert q5_answer >= 0 and q5_answer <= 1, "Probabilities should be between 0 and 1, inclusive."
        assert isinstance(q5_answer, float), "Probability should be a float accurate to 3 digits after the decimal point."
        assert abs(q5_answer - answer) < 1e-3, "Incorrect values."

        test_ok()

    @weight(5)
    def test_06(self):
        q6_answer = get_locals(self.notebook_locals, ["q6_answer"])
        answer = 0.613
        assert q6_answer >= 0 and q6_answer <= 1, "Probabilities should be between 0 and 1, inclusive."
        assert isinstance(q6_answer, float), "Probability should be a float accurate to 3 digits after the decimal point."
        assert abs(q6_answer - answer) < 1e-3, "Incorrect values."

        test_ok()

    @weight(2)
    def test_07(self):
        q7_answer = get_locals(self.notebook_locals, ["q7_answer"])
        answer = 8
        assert isinstance(q7_answer, int), "Can't have a decimal number of rows."
        assert q7_answer == answer, "Incorrect values."

        test_ok()

    @weight(4)
    def test_08(self):
        q8_answer = get_locals(self.notebook_locals, ["q8_answer"])
        answer = 0.0005
        assert q8_answer >= 0 and q8_answer <= 1, "Probabilities should be between 0 and 1, inclusive."
        assert isinstance(q8_answer, float), "Probability should be a float accurate to 3 digits after the decimal point."
        assert abs(q8_answer - answer) < 1e-5, "Incorrect values."
        
        test_ok()

    @weight(4)
    def test_09(self):
        q9_answer = get_locals(self.notebook_locals, ["q9_answer"])
        answer = 0.07781
        assert q9_answer >= 0 and q9_answer <= 1, "Probabilities should be between 0 and 1, inclusive."
        assert isinstance(q9_answer, float), "Probability should be a float accurate to 3 digits after the decimal point."
        assert abs(q9_answer - answer) < 1e-5, "Incorrect values."

        test_ok()

    @weight(5)
    def test_10(self):
        q10_answer = get_locals(self.notebook_locals, ["q10_answer"])
        answer = (0, 0.1, 0, 1)
        assert len(q10_answer) == len(answer), "Incorrect number of values."
        assert q10_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_11(self):
        q11_answer = get_locals(self.notebook_locals, ["q11_answer"])
        answer = (0.1, 1)
        assert len(q11_answer) == len(answer), "Incorrect number of values."
        assert q11_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_12(self):
        q12_answer = get_locals(self.notebook_locals, ["q12_answer"])
        answer = 4
        assert isinstance(q12_answer, int), "Please select only one option."
        assert q12_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_13(self):
        q13_answer = get_locals(self.notebook_locals, ["q13_answer"])
        answer = 3
        assert isinstance(q13_answer, int), "Please select only one option."
        assert q13_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_14_BN_warmup(self):
        BN_warmup = get_locals(
            self.notebook_locals, ["BN_warmup"])
        my_CPT = BN_warmup()
        assert isinstance(my_CPT.rvs, (tuple, list))
        assert len(my_CPT.rvs) == 2, "CPT should only have 2 RVs."
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
        assert abs(BN_warmup2(warmup2_CPT) - 0.01) < 1e-3, "Returned incorrect value"

        test_ok()


    @weight(6)
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


    @weight(4)
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
    @timeout_decorator.timeout(10.0)
    def test_19_minimal_bn(self):
        hidden_joint, q19_answer = get_locals(
            self.notebook_locals, ["hidden_joint", "q19_answer"])
        table = np.ascontiguousarray(hidden_joint.table, dtype=np.float64)
        assert hashlib.sha256(table.tobytes()).hexdigest() == _HIDDEN_JOINT_SHA256, \
            "hidden_joint has been changed. Reload it from the provided hidden_joint.npy."
        names = [rv.name for rv in hidden_joint.rvs]

        edges = _check_edge_list(q19_answer, names)
        order = _topological_order(names, edges)

        terms = _missing_dependence(table, names, edges, order)
        wrong = [x for x in order if terms[x] > _HIDDEN_JOINT_TOL]
        assert not wrong, (
            "Your Bayes net cannot represent hidden_joint: multiplying P(X | parents(X)) together "
            f"does not give back the joint (KL divergence {sum(terms.values()):.2e}). Going through the "
            f"variables in the order {order}, these depend on earlier variables beyond their parents: {wrong}.")
        assert len(edges) <= _HIDDEN_JOINT_MIN_EDGES, (
            f"Your Bayes net represents hidden_joint, but it has {len(edges)} edges; "
            f"one with {_HIDDEN_JOINT_MIN_EDGES} edges exists.")

        test_ok()


    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_20_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Pluto".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")