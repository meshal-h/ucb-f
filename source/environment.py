import numpy as np
import gymnasium as gym
from gymnasium import spaces


###################
### Environment ###
###################


class MDP(gym.Env):

    """
    Custom finite-horizon MDP with discrete state and action spaces:
        States evolve according to an additive disturbance model of the form s_next = ( f(s,a) + w(h) ) modulo S,
        modulo can be removed if the state space is allowed to vary for each h or if f(s,a) is restricted to be in [W,S-W].
    """

    metadata = {'render.modes': ['console']}
    
    def __init__(self, S, A, H, W):
        super(MDP, self).__init__()
        
        # State space, action space, and the horizon
        self.observation_space = spaces.Discrete(S)
        self.action_space = spaces.Discrete(A)
        self.horizon = H
                
        # Transition function f(s,a)
        self.F = np.random.randint(low=0, high=S, size=(S,A))
        self.F = np.repeat(self.F[np.newaxis, :, :], H, axis=0)
        
        # Transition randomness w(h)
        self.W = Disturbance(H, W)
        
        # Reward function r(s)
        self.R = np.random.rand(S)
        self.R = np.repeat(self.R[np.newaxis, :], H, axis=0)
        self.R = np.repeat(self.R[:, :, np.newaxis], A, axis=2)
        
        # Initial variables
        self.state = 0
        self.time = 0
        
        # To store the optimal Q-values and value functions
        self.Q_star = None
        self.V_star = None
        # To store the greedy policy w.r.t. the reward function
        self.V_greedy = None
        # To store the lipschitz constant
        self.L = None
        
    def reset(self):
        
        # Reset time
        self.time = 0
        
        # Initial state distribution (drawn from np.random so that a single seed controls all randomness)
        self.state = np.random.randint(self.observation_space.n)
        
        return self.state
    
    def step(self, action):
        
        # Transition function
        next_state = ( self.F[self.time, self.state, action] + self.W.sample(self.time) ) % self.observation_space.n
        
        # Reward function
        reward = self.R[self.time, self.state, action]
        
        # Termination condition
        done = not(self.time < self.horizon-1)
        
        # Additional information
        info = {}
        
        # Update state and time
        self.state = next_state
        self.time += 1
        
        return next_state, reward, done, info
    
    def render(self, mode='console'):
        pass
    
    def close (self):
        pass


class Disturbance:
    """ Discrete random variable with support in [0, maximum] and random probability mass function"""

    def __init__(self, horizon, maximum):
        
        self.min = 0
        self.max = maximum
        self.range = np.arange(0, self.max+1)
        
        if maximum > 0:
            self.prob = np.random.rand(horizon, self.max+1)
            self.prob = self.prob / np.sum(self.prob, axis=1).reshape(-1,1)
        else:
            self.prob = np.ones((horizon,1))
        
    def sample(self, h=0):
        
        return np.random.choice(a=self.range, p=self.prob[h])


###########################
### Dynamic Programming ###
###########################


def value_iteration(env):
    """Exact DP value iteration for the optimal Q-values function"""

    # Get problem cardinalities
    S = env.observation_space.n
    A = env.action_space.n
    H = env.horizon
    
    # Initiate Q-values
    Q = np.zeros((H,S,A))
    
    # Disturbance support
    w = env.W.range
    
    # Time loop; the state, action, and disturbance loops are vectorized
    for h in range(H-1,-1,-1):
        if h == H-1:
            Q[h] = env.R[h]
        else:
            V_next = np.max(Q[h+1], axis=1)                     # [S]
            s_next = ( env.F[h][:, :, np.newaxis] + w ) % S     # [S,A,W+1]
            Q[h] = env.R[h] + V_next[s_next] @ env.W.prob[h]    # [S,A]
    
    return Q


def policy_evaluation(env, Q):
    """Benchmark the greedy policy w.r.t given Q-values"""

    # Get problem cardinalities
    S = env.observation_space.n
    A = env.action_space.n
    H = env.horizon
    
    # Initiate value function
    V = np.zeros((H,S))
    
    # Disturbance support and state indices
    w = env.W.range
    states = np.arange(S)
    
    # Time loop; the state and disturbance loops are vectorized
    for h in range(H-1,-1,-1):
        # Greedy action for every state
        a = np.argmax(Q[h], axis=1)                              # [S]
        r = env.R[h, states, a]                                  # [S]
        if h == H-1:
            V[h] = r
        else:
            s_next = ( env.F[h, states, a][:, np.newaxis] + w ) % S   # [S,W+1]
            V[h] = r + V[h+1][s_next] @ env.W.prob[h]
    
    return V
