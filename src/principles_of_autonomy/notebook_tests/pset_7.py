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

class TestPSet7(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    def test_1(self):
        q1_answer = get_locals(self.notebook_locals, ["q1_answer"])
        answer = 2
        assert q1_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_2(self):
        q2_answer = get_locals(self.notebook_locals, ["q2_answer"])
        answer = "yes"
        assert q2_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_3(self):
        q3_answer = get_locals(self.notebook_locals, ["q3_answer"])
        answer = "no"
        assert q3_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_4(self):
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        answer = ('D', 'I')
        assert q4_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_5(self):
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        answer = ('B','F','G','H')
        assert q5_answer == answer, "Incorrect values."

        test_ok()


    @weight(5)
    def test_6(self):
        conditioning_warmup, RV, CPT = get_locals(self.notebook_locals, 
                                                  ["conditioning_warmup",
                                                   "RV", "CPT"])
        A = RV("A", [0, 1])
        B = RV("B", [0, 1])
        cpt = CPT(
          rvs=[A, B],
          table=np.array([
            [0.98, 0.01],
            [0.02, 0.99],
          ])
        )
        cond_cpt = conditioning_warmup(cpt, B)
        assert set(cond_cpt.rvs) == {A}, "Incorrect values."
        assert cond_cpt.get((0,)) == 0.98, "Incorrect values."
        assert cond_cpt.get((1,)) == 0.02, "Incorrect values."

        test_ok()

    @weight(5)
    def test_7(self):
        create_state_variable = get_locals(self.notebook_locals,
                                           ["create_state_variable"])
        map1 = [[0, 1, 0],
            [1, 0, 0]]
        rv = create_state_variable(map1, "current_state")
        assert rv.name == "current_state", "Incorrect value" 
        assert rv.dim == 4, "Incorrect value" 
        assert rv.domain == [(0, 0), (0, 2), (1, 1), (1, 2)], "Incorrect value" 

        test_ok()

    @weight(5)
    def test_8(self):
        create_state_variable, create_transition_cpt, CPT = \
            get_locals(self.notebook_locals, ["create_state_variable",
                                              "create_transition_cpt",
                                              "CPT"])
        map1 = [[0, 0, 0],
            [1, 0, 1]]
        s0 = create_state_variable(map1, "state_0")
        s1 = create_state_variable(map1, "state_1")
        cpt = create_transition_cpt(map1, s0, s1)
        assert set(cpt.rvs) == {s0, s1}, "State variables not set correctly"
        assert cpt.get_by_names({"state_0": (0, 0), "state_1": (0, 0)}) == \
            0.5, "Incorrect table values."
    
        assert cpt.get_by_names({"state_0": (0, 0), "state_1": (0, 1)}) == \
        0.5, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 0), "state_1": (1, 1)}) == \
        0., "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (1, 1)}) == \
        0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (0, 1)}) == \
        0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (0, 2)}) == \
        0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (0, 0)}) == \
        0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (1, 1), "state_1": (0, 0)}) == 0., \
        "Incorrect CPT variable names."
        assert cpt.get_by_names({"state_0": (1, 1), "state_1": (0, 1)}) == \
        0.5, "Incorrect CPT variable names."

        test_ok()

    @weight(5)
    def test_9(self):
        create_observation_variable, RV = \
        get_locals(self.notebook_locals, ["create_observation_variable", "RV"])

        rv = create_observation_variable("obs")
        assert rv.name == "obs"
        assert rv.dim == 2 ** 4
        assert set(rv.domain) == {
            (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0),
            (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0), (1, 1, 1, 0),
            (0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1),
            (0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1),
        }

        test_ok()

    @weight(5)
    def test_10(self):
        create_state_variable, RV, create_observation_variable, \
        create_observation_cpt = \
        get_locals(self.notebook_locals, ["create_state_variable", "RV",
                                          "create_observation_variable",
                                          "create_observation_cpt"])

        map1 = [[0, 0, 0],
            [1, 0, 1]]
        s0 = create_state_variable(map1, "state_0")
        z0 = create_observation_variable("obs_0")
        cpt = create_observation_cpt(map1, s0, z0)
        assert set(cpt.rvs) == {s0, z0}, "Incorrect RVs in the CPT"
        assert cpt.get_by_names({"state_0": (0, 0), "obs_0": (1, 0, 1, 1)}) ==\
                1., "Incorrect value in the CPT"
        assert cpt.get_by_names({"state_0": (0, 0), "obs_0": (1, 1, 0, 1)}) ==\
                0., "Incorrect value in the CPT"
        assert cpt.get_by_names({"state_0": (0, 1), "obs_0": (1, 0, 0, 0)}) ==\
                1., "Incorrect value in the CPT"
        assert cpt.get_by_names({"state_0": (0, 2), "obs_0": (1, 1, 1, 0)}) ==\
                1., "Incorrect value in the CPT"
        assert cpt.get_by_names({"state_0": (1, 1), "obs_0": (0, 1, 1, 1)}) ==\
                1., "Incorrect value in the CPT"

        test_ok()

                
    @weight(10)
    def test_11(self):
        create_hmm, run_viterbi = get_locals(self.notebook_locals,
                                             ["create_hmm", "run_viterbi"])

        map1 = [[0, 0, 0],
                [1, 0, 1]]
        observations1 = [(1, 0, 1, 1), (1, 0, 0, 0), (1, 0, 0, 0)]
        hmm1 = create_hmm(map1, len(observations1))
        most_likely_states1 = run_viterbi(hmm1, observations1)
        assert most_likely_states1 == [(0, 0), (0, 1), (0, 1)]

        map2 = [[0, 0, 0, 0, 0]]
        observations2 = [(1, 0, 1, 0), (1, 0, 1, 1), (1, 0, 1, 0),
                         (1, 0, 1, 0), (1, 0, 1, 0), (1, 1, 1, 0)]
        hmm2 = create_hmm(map2, len(observations2))
        most_likely_states2 = run_viterbi(hmm2, observations2)
        assert most_likely_states2 == [(0, 1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

        map3 = [[1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 1, 0, 1]]
        observations3 = [(0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 0)]
        hmm3 = create_hmm(map3, len(observations3), noise_prob=0.05)
        most_likely_states3 = run_viterbi(hmm3, observations3)
        assert most_likely_states3 == [(2, 3), (2, 3), (1, 3), (1, 4)]

        map4 = [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ]
        observations4 = [(1, 0, 1, 1), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 0, 0)]
        hmm4 = create_hmm(map4, len(observations4), noise_prob=0.05)
        most_likely_states4 = run_viterbi(hmm4, observations4)
        assert most_likely_states4 == [(3, 0), (3, 1), (3, 2), (3, 3)]

        test_ok()
        
    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_19_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Greece".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
