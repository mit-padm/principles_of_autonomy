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


# ---------------------------------------------------------------------------
# Problem 3A: a Bayes net that defeats rejection sampling.
# Conditions 1-4 are checked exactly (enumerating at most 2^10 joint
# assignments); condition 5 (P(e) <= 0.005) is checked empirically with the
# grader's own forward sampler. With N = 8000 and a cutoff of 80 accepts
# (2x the expected 40 accepts at P(e) = 0.005), a net with P(e) <= 0.005 fails
# with probability < 1e-8, and one with P(e) >= 0.02 passes with prob. < 1e-11.
# ---------------------------------------------------------------------------
_RS_MAX_VARS = 10
_RS_CPT_MIN, _RS_CPT_MAX = 0.05, 0.95
_RS_MAX_EVIDENCE = 3
_RS_MIN_MARGINAL = 0.5
_RS_NUM_SAMPLES = 8000
_RS_MAX_ACCEPTS = 80
_RS_SEED = 9
_RS_TOL = 1e-9


def _rs_check_net(cpts):
    """Validate a list of CPTs (parents first, child last) over binary RVs.

    Returns (names, parents, tables): names in a topological order, a dict
    from name to list of parent names, and a dict from name to a float array
    of shape (2,) * (num_parents + 1) indexed [parent values..., child value].
    """
    assert isinstance(cpts, (list, tuple)), "cpts must be a list of CPTs."
    assert 1 <= len(cpts) <= _RS_MAX_VARS, \
        f"Your net has {len(cpts)} variables; it must have between 1 and {_RS_MAX_VARS}."

    parents, tables, given_order = {}, {}, []
    for cpt in cpts:
        rvs = list(cpt.rvs)
        assert len(rvs) >= 1, "Every CPT must contain at least one RV."
        for rv in rvs:
            assert list(rv.domain) == [0, 1], \
                f"Variable {rv.name} has domain {rv.domain}; every variable must have domain [0, 1]."
        names = [rv.name for rv in rvs]
        child = names[-1]
        assert len(set(names)) == len(names), f"The CPT for {child} lists a variable more than once."
        assert child not in tables, f"Variable {child} is the last (child) variable of more than one CPT."
        table = np.asarray(cpt.table, dtype=np.float64)
        assert table.shape == (2,) * len(rvs), \
            f"The CPT for {child} has table shape {table.shape}; expected {(2,) * len(rvs)}."
        assert np.all(np.isfinite(table)), f"The CPT for {child} contains non-finite entries."
        assert table.min() >= _RS_CPT_MIN - _RS_TOL and table.max() <= _RS_CPT_MAX + _RS_TOL, (
            f"The CPT for {child} has entries in [{table.min():.4g}, {table.max():.4g}]; "
            f"every entry must be in [{_RS_CPT_MIN}, {_RS_CPT_MAX}].")
        assert np.allclose(table.sum(axis=-1), 1.0, atol=1e-6), (
            f"The CPT for {child} does not sum to 1 over its last axis. Remember to list "
            "the parents first and the variable itself last.")
        parents[child] = names[:-1]
        tables[child] = table
        given_order.append(child)

    for child, pas in parents.items():
        for pa in pas:
            assert pa in tables, \
                f"{pa} is a parent of {child} but has no CPT of its own (no CPT lists it last)."

    # Kahn's algorithm, breaking ties by the order the CPTs were given in.
    order, placed = [], set()
    while len(order) < len(given_order):
        ready = [x for x in given_order if x not in placed and all(p in placed for p in parents[x])]
        assert ready, "Your net has a directed cycle."
        order.append(ready[0])
        placed.add(ready[0])
    return order, parents, tables


def _rs_check_evidence(evidence, names):
    assert isinstance(evidence, dict), "evidence must be a dict from variable names to values."
    assert 1 <= len(evidence) <= _RS_MAX_EVIDENCE, \
        f"Your evidence assigns {len(evidence)} variables; it must assign between 1 and {_RS_MAX_EVIDENCE}."
    clean = {}
    for name, value in evidence.items():
        assert name in names, f"Evidence variable {name!r} is not a variable in your net."
        assert value in (0, 1), f"Evidence value for {name} is {value!r}; it must be 0 or 1."
        clean[name] = int(value)
    return clean


def _rs_joint(order, parents, tables):
    """Exact joint as an array with one binary axis per variable in `order`."""
    operands = []
    for x in order:
        operands.append(tables[x])
        operands.append([order.index(v) for v in parents[x] + [x]])
    return np.einsum(*operands, list(range(len(order))))


def _rs_forward_sample(order, parents, tables, num_samples, seed):
    """Forward-sample the net. Returns an int array of shape (num_samples, len(order))."""
    rng = np.random.default_rng(seed)
    col = {x: i for i, x in enumerate(order)}
    samples = np.zeros((num_samples, len(order)), dtype=np.int64)
    for x in order:
        index = tuple(samples[:, col[p]] for p in parents[x]) + (1,)
        p_one = tables[x][index]
        samples[:, col[x]] = (rng.random(num_samples) < p_one).astype(np.int64)
    return samples


class TestPSet9(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    def test_01(self):
        q1_answer = get_locals(self.notebook_locals, ["q1_answer"])
        answer = 2
        assert q1_answer == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_02(self):
        q2_answer = get_locals(self.notebook_locals, ["q2_answer"])
        answer = "yes"
        assert q2_answer.strip().lower() == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_03(self):
        q3_answer = get_locals(self.notebook_locals, ["q3_answer"])
        answer = "no"
        assert q3_answer.strip().lower() == answer, "Incorrect values."

        test_ok()

    @weight(5)
    def test_04(self):
        q4_answer = get_locals(self.notebook_locals, ["q4_answer"])
        answer = ('D', 'I')
        assert len(q4_answer) == len(answer), "Incorrect number of values."
        assert set(q4_answer) == set(answer), "Incorrect values."

        test_ok()

    @weight(5)
    def test_05(self):
        q5_answer = get_locals(self.notebook_locals, ["q5_answer"])
        answer = ('B','E','G','H')
        assert len(q5_answer) == len(answer), "Incorrect number of values."
        assert set(q5_answer) == set(answer), "Incorrect values."

        test_ok()


    @weight(5)
    def test_06(self):
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
        assert set(cond_cpt.rvs) == {A}, "Incorrect RVs in CPT."
        assert cond_cpt.get((0,)) == 0.98, "Incorrect values."
        assert cond_cpt.get((1,)) == 0.02, "Incorrect values."

        test_ok()

    @weight(5)
    def test_07(self):
        create_state_variable = get_locals(self.notebook_locals,
                                           ["create_state_variable"])
        map1 = [[0, 1, 0],
            [1, 0, 0]]
        rv = create_state_variable(map1, "current_state")
        assert rv.name == "current_state", "Incorrect RV name." 
        assert rv.dim == 4, "Incorrect number of values in RV's domain." 
        assert rv.domain == [(0, 0), (0, 2), (1, 1), (1, 2)], "Incorrect RV domain." 

        test_ok()

    @weight(10)
    def test_08(self):
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
        assert cpt.get_by_names({"state_0": (0, 0), "state_1": (0, 0)}) == 0.5, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 0), "state_1": (0, 1)}) == 0.5, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 0), "state_1": (1, 1)}) == 0., "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (1, 1)}) == 0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (0, 1)}) == 0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (0, 2)}) == 0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (0, 1), "state_1": (0, 0)}) == 0.25, "Incorrect table values."
        assert cpt.get_by_names({"state_0": (1, 1), "state_1": (0, 0)}) == 0., "Incorrect table values."
        assert cpt.get_by_names({"state_0": (1, 1), "state_1": (0, 1)}) == 0.5, "Incorrect table values."

        test_ok()

    @weight(5)
    def test_09(self):
        create_observation_variable, RV = \
        get_locals(self.notebook_locals, ["create_observation_variable", "RV"])

        rv = create_observation_variable("obs")
        assert rv.name == "obs", "Incorrect RV name."
        assert rv.dim == 2 ** 4, "Incorrect number of values in RV's domain."
        assert set(rv.domain) == {
            (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0),
            (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0), (1, 1, 1, 0),
            (0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1),
            (0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1),
        }, "Incorrect RV domain."

        test_ok()

    @weight(25)
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

                
    @weight(25)
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
    def test_12_rejection_adversary(self):
        rejection_sampling_adversary = get_locals(
            self.notebook_locals, ["rejection_sampling_adversary"])
        result = rejection_sampling_adversary()
        assert isinstance(result, tuple) and len(result) == 2, \
            "rejection_sampling_adversary() must return a tuple (cpts, evidence)."
        cpts, evidence = result

        # Conditions 1-3.
        order, parents, tables = _rs_check_net(cpts)
        evidence = _rs_check_evidence(evidence, order)

        # Condition 4, exactly.
        joint = _rs_joint(order, parents, tables)
        for name, value in evidence.items():
            axis = order.index(name)
            other_axes = tuple(i for i in range(len(order)) if i != axis)
            marginal = joint.sum(axis=other_axes)[value]
            assert marginal >= _RS_MIN_MARGINAL - _RS_TOL, (
                f"P({name} = {value}) = {marginal:.4f}, but every evidence value must have "
                f"probability at least {_RS_MIN_MARGINAL} on its own.")

        # Condition 5, empirically: rejection sampling with the grader's own sampler.
        samples = _rs_forward_sample(order, parents, tables, _RS_NUM_SAMPLES, _RS_SEED)
        ev_cols = [order.index(name) for name in evidence]
        ev_vals = np.array([evidence[name] for name in evidence])
        accepted = np.all(samples[:, ev_cols] == ev_vals, axis=1)
        num_accepted = int(accepted.sum())

        print(f"Rejection sampling with evidence {evidence}: accepted {num_accepted} / {_RS_NUM_SAMPLES} samples "
              f"(empirical acceptance rate {num_accepted / _RS_NUM_SAMPLES:.4f}); "
              f"passing requires at most {_RS_MAX_ACCEPTS}.")

        assert num_accepted <= _RS_MAX_ACCEPTS, (
            f"Rejection sampling accepted {num_accepted} of {_RS_NUM_SAMPLES} samples; "
            f"at most {_RS_MAX_ACCEPTS} are allowed. Make the evidence jointly less likely.")

        test_ok()

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_13_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Uran".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")