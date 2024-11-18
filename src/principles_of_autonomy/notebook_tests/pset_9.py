import unittest
import timeout_decorator
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight
# from nose.tools import assert_equal

from principles_of_autonomy.grader import get_locals, compare_iterators


class TestPSet9(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(2)
    def test_01(self):
        t = (True, False, False, False)
        q1_answer = get_locals(self.notebook_locals, ["q1_answer"])
        assert compare_iterators(q1_answer, t)

    @weight(2)
    def test_02(self):
        t = (True, False, False, True)
        q2_answer = get_locals(self.notebook_locals, ["q2_answer"])
        assert compare_iterators(q2_answer, t)

    @weight(2)
    def test_03(self):
        t = (True, False, True, False)
        q3_answer = get_locals(self.notebook_locals, ["q3_answer"])
        assert compare_iterators(q3_answer, t)

    @weight(2)
    def test_04(self):
        t = (True, True, False, False)
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        assert compare_iterators(q4_answer, t)

    @weight(2)
    def test_05(self):
        t = (False, False, False, True)
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        assert compare_iterators(q5_answer, t)

    # 1.2
    @weight(2)
    def test_06(self):
        t = [-3, -5]
        obj_simple = get_locals(self.notebook_locals, ["obj_simple"])
        assert compare_iterators(obj_simple, t)

    @weight(2)
    def test_07(self):
        t = [[1, 0], [0, 2], [3, 2]]
        lhs_simple = get_locals(self.notebook_locals, ["lhs_simple"])
        assert compare_iterators(lhs_simple, t)

    @weight(2)
    def test_08(self):
        t = [4, 12, 18]
        rhs_simple = get_locals(self.notebook_locals, ["rhs_simple"])
        assert compare_iterators(rhs_simple, t)

    @weight(4)
    def test_09(self):
        dec_vars, opt = get_locals(self.notebook_locals, [
                                   "decision_variables_simple", "optimum_simple"])
        dec_vars_ans = [2., 6.]
        opt_ans = 36
        assert np.allclose(dec_vars, dec_vars_ans)
        assert np.allclose(opt, opt_ans)

    # 1.3
    @weight(2)
    def test_10(self):
        obj = get_locals(self.notebook_locals, ["obj_image"])
        t = [-20, -12, -30, -15]
        assert compare_iterators(t, obj)

    @weight(2)
    def test_11(self):
        lhs = get_locals(self.notebook_locals, ["lhs_image"])
        t = [[1, 1, 1, 1], [3, 2, 2, 0], [0, 1, 5, 3]]
        assert compare_iterators(t, lhs)

    @weight(2)
    def test_12(self):
        rhs = get_locals(self.notebook_locals, ["rhs_image"])
        t = [50, 100, 100]
        assert compare_iterators(t, rhs)

    @weight(4)
    def test_13(self):
        lp_soln = get_locals(self.notebook_locals, ["lp_soln_image"])
        x_soln = [25., 0., 12.5, 12.5]
        fun_soln = 1062.5
        assert np.allclose(lp_soln.x, x_soln)
        assert np.allclose(-lp_soln.fun, fun_soln)

    @weight(2)
    def test_14(self):
        image2 = get_locals(self.notebook_locals, ["image2"])
        image2_soln = 0
        assert np.allclose(image2, image2_soln)

    @weight(2)
    def test_15(self):
        utility = get_locals(self.notebook_locals, ["utility_image"])
        utility_soln = 1062.5
        assert np.allclose(utility, utility_soln)

    @weight(2)
    def test_16(self):
        max_utility_camera = get_locals(
            self.notebook_locals, ["max_utility_camera"])
        max_utility_soln = 0
        assert np.allclose(max_utility_camera, max_utility_soln)

    @weight(4)
    def test_17(self):
        lp_soln_full_image = get_locals(
            self.notebook_locals, ["lp_soln_full_image"])
        x_ans = [23.3333, 0., 11.5, 11.1666]
        fun_ans = 979.1666
        assert np.allclose(lp_soln_full_image.x, x_ans, rtol=1e-3)
        assert np.allclose(-lp_soln_full_image.fun, fun_ans, rtol=1e-3)

    @weight(2)
    def test_18(self):
        utility_new = get_locals(self.notebook_locals, ["utility_new"])
        u_ans = 1056.1666
        assert np.allclose(utility_new, u_ans, rtol=1e-3)

    @weight(2)
    def test_19(self):
        change = get_locals(self.notebook_locals, ["change_image"])
        change_ans = 'down'
        assert hash(change.strip().lower()) == hash(change_ans)

    # 2.1
    @weight(4)
    def test_20(self):
        lp_soln_knapsack_no_enforce = get_locals(
            self.notebook_locals, ["lp_soln_knapsack_no_enforce"])
        x_ans = [0, 1, 1, 1, 0]
        fun_ans = 32
        assert np.allclose(lp_soln_knapsack_no_enforce.x, x_ans, rtol=1e-3)
        assert np.allclose(-lp_soln_knapsack_no_enforce.fun,
                           fun_ans, rtol=1e-3)

    @weight(2)
    def test_21(self):
        num_items = get_locals(self.notebook_locals, ["num_items"])
        ans = 3
        assert num_items == ans

    @weight(2)
    def test_22(self):
        weight_initial = get_locals(self.notebook_locals, ["weight_initial"])
        ans = 15
        assert np.allclose(weight_initial, ans, rtol=1e-3)

    @weight(4)
    def test_23(self):
        lp_soln_updated = get_locals(self.notebook_locals, ["lp_soln_updated"])
        x_ans = [1, 1, 1, 0.375, 0]
        fun_ans = 31.25
        assert np.allclose(lp_soln_updated.x, x_ans, rtol=1e-3)
        assert np.allclose(-lp_soln_updated.fun, fun_ans, rtol=1e-3)

    @weight(2)
    def test_24(self):
        weight_no_longer_linear = get_locals(
            self.notebook_locals, ["weight_no_longer_linear"])
        ans = 15
        assert np.allclose(weight_no_longer_linear, ans, rtol=1e-3)

    @weight(2)
    def test_25(self):
        value_no_longer_linear = get_locals(
            self.notebook_locals, ["value_no_longer_linear"])
        ans = 31.25
        assert np.allclose(value_no_longer_linear, ans, rtol=1e-3)

    # 2.2
    @weight(2)
    def test_26(self):
        b_l = get_locals(self.notebook_locals, ["b_l_knapsack"])
        ans = np.all(np.isinf(np.array(b_l)))
        assert ans

    @weight(2)
    def test_27(self):
        constraints_knapsack = get_locals(
            self.notebook_locals, ["constraints_knapsack"])
        lhs = [[4, 5, 3, 8, 6],
               [1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]
        rhs = [15, 1, 1, 1, 1, 1]
        assert np.array_equal(constraints_knapsack.A, lhs)
        assert np.array_equal(constraints_knapsack.ub, rhs)

    @weight(2)
    def test_28(self):
        integrality_knapsack = get_locals(
            self.notebook_locals, ["integrality_knapsack"])
        obj = [-8, -12, -6, -14, -10]
        t = np.ones_like(obj)
        assert np.array_equal(integrality_knapsack, t)

    @weight(4)
    def test_29(self):
        ip_soln_knapsack = get_locals(
            self.notebook_locals, ["ip_soln_knapsack"])
        x_soln = [1, 1, 0, 0, 1]
        fun_soln = 30
        assert np.array_equal(ip_soln_knapsack.x, x_soln)
        assert np.allclose(-ip_soln_knapsack.fun, fun_soln)
    @weight(2)
    def test_30(self):
        weight_ip = get_locals(
            self.notebook_locals, ["weight_ip"])
        s = 15
        assert np.allclose(weight_ip, s, rtol=1e-3)

    @weight(2)
    def test_31(self):
        change_ip = get_locals(self.notebook_locals, ["change_ip"])
        change_ans = 'worse'
        assert hash(change_ip.strip().lower()) == hash(change_ans)


    #### 3.1
    @weight(8)
    def test_32(self):
        image_allocation = get_locals(self.notebook_locals, ["image_allocation"])
        a = [24, 1, 12, 13]
        assert np.allclose(image_allocation, a, rtol=1e-3)

    @weight(8)
    def test_33(self):
        image_allocation_relaxed = get_locals(self.notebook_locals, ["image_allocation_relaxed"])
        a = [24.2666, 1, 12.6, 12]
        assert np.allclose(image_allocation_relaxed, a, rtol=1e-3)

    @weight(10)
    def test_34(self):
        image_allocation_multiple = get_locals(self.notebook_locals, ["image_allocation_multiple"])
        a = [36, 1, 12, 1]
        assert np.allclose(image_allocation_multiple, a, rtol=1e-3)

    @weight(2)
    def test_35(self):
        process_location = get_locals(self.notebook_locals, ["process_location"])
        first = 'gpu'
        second = ['cpu', 'gpu']
        assert process_location[0].lower() == first and process_location[1].lower() in second

    @weight(2)
    def test_36(self):
        utility_multiple = get_locals(self.notebook_locals, ["utility_multiple"])
        a = 1107
        assert np.allclose(a, utility_multiple)

    @weight(2)
    def test_37(self):
        alt, orig = get_locals(self.notebook_locals, ["alternate_solution", "process_location"])
        assert alt is not None
        assert not compare_iterators(alt, orig), "Same solution as before is not alternate"
        first = 'gpu'
        second = ['cpu', 'gpu']
        assert alt[0].lower() == first and alt[1].lower() in second

    @weight(6)
    def test_38(self):
        beer_order = get_locals(self.notebook_locals, ["beer_order"])
        a = 2000
        b = 1000
        assert np.allclose(beer_order['regular'], a)
        assert np.allclose(beer_order['strong'], b)

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_99_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("simplex".lower())
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
