import unittest
import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
# from nose.tools import assert_equal

from principles_of_autonomy.grader import get_locals
import random
import numpy as np
import random

class MDP(object):
    """Simple MDP class."""
    def __init__(self, S, A, T, R, gamma):
        """Define MDP"""
        self.S = S # Set of states
        self.A = A # Set of actions
        self.T = T # Transition probabilities: T[s][a][s']
        self.R = R # Rewards: R[s][a][s']
        self.gamma = gamma # Discount factor
    def print_mdp(self):
        # Code to print MDP
        print("MDP: \n  States (%d): %s\n  Actions (%d): %s" % (len(self.S),
                                                           ", ".join(["'%s'" % str(s) for s in self.S])
                                                                ,len(self.A)                                                                
                                                                , self.A))
        print("  Transitions:")
        for sij in sorted(self.S):
            print("   + State '%s'" % (str(sij)))
            for a in self.T[sij]:
                print("     -action '%s'" % str(a))
                for sdest,pdest in self.T[sij][a].items():
                    print("       to '%s' with p=%.2f" %(str(sdest), pdest))
        print("  Rewards:") 
        for sij in sorted(self.S):
            if sij in self.R:
                print("    + from state '%s'" % str(sij))
                for a in self.A:
                    if a in self.R[sij]:
                        for sdest,rdest in self.R[sij][a].items():
                            if not rdest == 0:
                                print("      - with '%s' to state '%s', r=%.2f" % (a, sdest, rdest))
    def is_sink_state(self, state):
        """
        Check if a state is a sink state.
        A sink state is defined as a state where all actions lead back to the same state.
        
        Parameters:
        - state: The state to check
        
        Returns:
        - True if the state is a sink state, False otherwise.
        """
        for action in self.T[state]:
            if not (len(self.T[state][action]) == 1 and state in self.T[state][action] and self.T[state][action][state] == 1.0):
                return False
        return True

def build_mdp(n, p, obstacles, goal, gamma, goal_reward=100, obstacle_reward=-1000):
    S = set()
    T = dict()
    R = dict()
    actions = ['up','down','right','left']
    vertical = [(0,1), (0,-1)]
    horizontal = [(1,0), (-1,0)]
    directions =  vertical + horizontal
    right_angles = [horizontal, horizontal, vertical, vertical]
    action_dest_dirs = [[directions[i]] + right_angles[i] for i in range(len(actions))]
    
    def xy_to_i(x,y):
        return n*x + y
    def apply_direction(x,y,direction):
        return (x+direction[0], y+direction[1])
    def valid_state(x,y):
        return 0<=x<n and 0<=y<n
    def neighbors(x,y):
        candidates = [(x+dx, y+dy) for dx,dy in directions]
        coords = (x,y)
        return list(filter(lambda coords: valid_state(coords[0],coords[1]), candidates))    
    def action_state(x, y, a):
        dx, dy = directions[actions.index(a)]
        return (x+dx, y+dy)
    def action_dest_states(x, y, a):
        dx, dy = directions[actions.index(a)]
        candidate_states = [apply_direction(x,y,d) for d in action_dest_dirs[actions.index(a)]]
        # If hitting obstacle, bounce back to same state
        return [cs if valid_state(*cs) else (x,y) for cs in candidate_states ]
        
    # Add states and transitions
    for i in range(n):
        for j in range(n):            
            sij = (i, j)
            S.add(sij)
            T[sij] = dict()
            ij_neighbors = neighbors(i,j)
            for ai, a in enumerate(actions):
                T[sij][a] = dict()
                dest_states = action_dest_states(i,j, a)
                T[sij][a][dest_states[0]] = p # main outcome of action
                remaining_p = (1-p)/2.0
                for other_s in dest_states[1:]:
                    T[sij][a][other_s] = T[sij][a].get(other_s, 0.0) + remaining_p
                assert np.allclose(sum([p_s_sdest for p_s_sdest in T[sij][a].values()]),1.0),\
                            "The sum of p for state %s should be 1.0 but it's %.2f" %(str(sij),
                                                                                         sum([p_s_sdest for p_s_sdest in T[sij][a].values()]))

            # Reward function
            # R(s, s')
            # Reward is 0 for all neighbor nodes by default
            R[sij] = dict()
            for a in actions:
                R[sij][a] = dict()
                for sdest in ij_neighbors:
                    R[sij][a][sdest] = 0.0
    

    # Add reward for goal    
    for nn in neighbors(*goal):
        for a in actions:
            if goal in T[nn][a]:
                R[nn][a][goal] = goal_reward
    
    # Negative rewards for obstacles
    for obs in obstacles:
        for a in actions:
            for nn in neighbors(*obs):
                if obs in T[nn][a]: 
                    R[nn][a][obs] = obstacle_reward
    
    # Make goal and obstacles sink states
    for sink_s in [goal]+obstacles:
        for a in actions:
            T[sink_s][a] = {sink_s: 1.0}
                    
                    
    mdp = MDP(S, actions, T, R, gamma)
    return mdp

def sample_environment(mdp, state, action):
    """
    Simulates the environment's response to taking an action in a given state.
    
    Args:
        mdp (MDP): The MDP object (with transition and reward functions).
        state: The current state.
        action: The action taken.
        
    Returns:
        next_state: The resulting next state after taking the action.
        reward: The reward received after transitioning to the next state.
    """
    # Sample the next state based on transition probabilities
    next_state = random.choices(list(mdp.T[state][action].keys()), 
                                weights=mdp.T[state][action].values())[0]
    
    # Get the reward for transitioning to the next state
    reward = mdp.R[state][action].get(next_state, 0.0)
    
    return next_state, reward

def generate_episodes(mdp, policy, num_episodes=5, max_steps=15, random_seed=None):
    """
    Generates a list of episodes following the given policy in the specified MDP.

    Parameters:
    - mdp: An instance of the MDP class
    - policy: A dictionary that maps states to actions
    - num_episodes: The number of episodes to generate
    - max_steps: Maximum number of steps before termination in each episode

    Returns:
    - episodes: A list of episodes, where each episode is a list of (state, action, reward) tuples
    """
    if random_seed:
        random.seed(random_seed)   # Set random seed for reproducibility
    episodes = []

    for _ in range(num_episodes):
        episode = []
        state = random.choice(list(mdp.S))  # Start from a random state

        for _ in range(max_steps):
            action = policy[state]  # Select action according to policy
            next_state, reward = sample_environment(mdp, state, action)
            episode.append((state, action, next_state, reward))

            state = next_state
            
            if mdp.is_sink_state(state):
                # Episode ends if we reach a terminal state or sink state
                break

        episodes.append(episode)

    return episodes
# Function for tests
def test_ok():
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Test passed!!</strong>
        </div>""", raw=True)
    except:
        print("test ok!!")

class TestPSet5(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(20)
    @timeout_decorator.timeout(120.0)
    def test_1_direct_evaluation(self):
        direct_evaluation = get_locals(
            self.notebook_locals, ["direct_evaluation"])
        n = 3
        goal = (2, 2)
        obstacles = [(0, 1)]
        mdp = build_mdp(n, 0.8, obstacles, goal, 0.8)

        # Create simple policy that always moves right.
        policy = {s: 'right' for s in mdp.S}

        # Generate some episodes.
        episodes = generate_episodes(mdp, policy, num_episodes=100000, max_steps=100, random_seed = 100)

        # Perform direct evaluation with these episodes.
        V = direct_evaluation(episodes, mdp.gamma)

        # Expected values for each state
        expected_values = {
        (0, 0): -101.67,
        (0, 1): 0.0,    # Obstacle
        (0, 2): -44.78,
        (1, 0): 8.19,
        (1, 1): 26.41,
        (1, 2): 89.34,
        (2, 0): 8.5,
        (2, 1): 29.7,
        (2, 2): 0.0,    # Goal
        }

        
        # Assert that the computed values match the expected values
        for state, expected_value in expected_values.items():
            assert abs(V[state] - expected_value) < 0.1, "Incorrect values."

        test_ok()

    @weight(20)
    @timeout_decorator.timeout(120.0)
    def test_2_TD_learning(self):
        td_learning = get_locals(
            self.notebook_locals, ["td_learning"])
        n = 3
        goal = (2, 2)
        obstacles = [(0, 1)]
        mdp = build_mdp(n, 0.8, obstacles, goal, 0.8)

        # Create simple policy that always moves right.
        policy = {s: 'right' for s in mdp.S}

        # Generate some episodes.
        episodes = generate_episodes(mdp, policy, num_episodes=100000, max_steps=100, random_seed = 100)

        # Perform td learning with these episodes.
        alpha = 0.01
        V = td_learning(episodes, mdp.gamma, alpha)

        # Expected values for each state
        expected_values = {
            (0, 0): -135.1,
            (0, 1): 0.0,    # Obstacle
            (0, 2): -43.98,
            (1, 0): 8.07,
            (1, 1): 28.07,
            (1, 2): 88.11,
            (2, 0): 9.08,
            (2, 1): 33.92,
            (2, 2): 0.0,    # Goal
        }

        
        # Assert that the computed values match the expected values
        for state, expected_value in expected_values.items():
            assert abs(V[state] - expected_value) < 0.1, "Incorrect values."

        test_ok()

    @weight(20)
    @timeout_decorator.timeout(120.0)
    def test_3_Q_learning(self):
        random.seed(0)
        q_learning = get_locals(
            self.notebook_locals, ["q_learning"])
     # Build MDP with p=0.8 and gamma=0.8. Use default rewards.
        n = 3
        goal = (2, 2)
        obstacles = [(0, 1)]
        mdp = build_mdp(n, 0.8, obstacles, goal, 0.8)

        # Perform Q learning with these episodes.
        alpha = 0.01
        num_episodes = 10000
        Q, policy = q_learning(num_episodes, mdp, mdp.gamma, alpha)

        # Expected values for each state
        Q_expected_values = {
        (0, 1): {'up': 0.0, 'down': 0.0, 'right': 0.0, 'left': 0.0},
        (1, 2): {'up': 72.82, 'down': 53.39, 'right': 93.46, 'left': 30.26},
        (2, 1): {'up': 93.26, 'down': 57.8, 'right': 76.14, 'left': 61.39},
        (0, 0): {'up': -822.43, 'down': 15.55, 'right': -49.0, 'left': -86.66},
        (1, 1): {'up': -45.43, 'down': -60.75, 'right': 72.76, 'left': -837.7},
        (2, 0): {'up': 69.65, 'down': 54.77, 'right': 57.53, 'left': 48.06},
        (0, 2): {'up': 25.01, 'down': -779.06, 'right': -49.46, 'left': -91.05}
        }

        policy_expected_values = {(1, 1): 'right',
        (2, 0): 'up',
        (0, 2): 'up',
        (2, 2): 'up',
        (1, 0): 'right'}

        # Assert that the computed values match the expected values
        for state, expected_value in Q_expected_values.items():
            for action, value in expected_value.items():
                assert abs(Q[state][action] - value) < 0.1, "Incorrect values for Q."

        # Assert that the computed values match the expected values
        for state, expected_value in policy_expected_values.items():
            assert policy[state]==expected_value, "Incorrect policy."

        test_ok()

    @weight(10)
    @timeout_decorator.timeout(120.0)
    def test_4_Q_learning(self):
        q_learning_epsilon_greedy = get_locals(
            self.notebook_locals, ["q_learning_epsilon_greedy"])
        # Build MDP with p=0.8 and gamma=0.8. Use default rewards.
        n = 3
        goal = (2, 2)
        obstacles = [(0, 1)]
        mdp = build_mdp(n, 0.8, obstacles, goal, 0.8)

        # Perform Q learning with these episodes.
        alpha = 0.01
        epsilon = 0.5
        num_episodes = 10000
        random.seed(0)
        Q, policy = q_learning_epsilon_greedy(num_episodes, mdp, mdp.gamma, alpha, epsilon)

        # Expected values for each state
        Q_expected_values = {
        (0, 1): {'up': 0.0, 'down': 0.0, 'right': 0.0, 'left': 0.0},
        (1, 2): {'up': 69.46, 'down': 58.35, 'right': 93.76, 'left': 29.47},
        (2, 1): {'up': 94.61, 'down': 57.37, 'right': 74.61, 'left': 61.44},
        (0, 0): {'up': -772.95, 'down': 17.22, 'right': -67.04, 'left': -73.13},
        (1, 1): {'up': -3.79, 'down': -47.33, 'right': 70.83, 'left': -773.34},
        (2, 0): {'up': 69.15, 'down': 54.22, 'right': 57.75, 'left': 48.29},
        (0, 2): {'up': 26.21, 'down': -780.78, 'right': -22.68, 'left': -47.94},
        (2, 2): {'up': 0.0, 'down': 0.0, 'right': 0.0, 'left': 0.0},
        (1, 0): {'up': 52.43, 'down': 40.94, 'right': 53.83, 'left': 18.69}
        }

        policy_expected_values =  {(0, 1): 'up',
        (1, 2): 'right',
        (2, 1): 'up',
        (0, 0): 'down',
        (1, 1): 'right',
        (2, 0): 'up',
        (0, 2): 'up',
        (2, 2): 'up',
        (1, 0): 'right'}

        
        # Assert that the computed values match the expected values
        for state, expected_value in Q_expected_values.items():
            for action, value in expected_value.items():
                assert abs(Q[state][action] - value) < 0.1, "Incorrect values for Q."

        # Assert that the computed values match the expected values
        for state, expected_value in policy_expected_values.items():
            assert policy[state]==expected_value, "Incorrect policy."

        test_ok()


    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_5_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Elk".lower()) #to change!!
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")