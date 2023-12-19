import numpy as np
from .environment import *
from .algorithm import *


#############################################
### Main code for running the experiments ###
#############################################


def calculate_regret(env, K, algorithm, arg={}):
    
    """
    Main code to calculate regret.
    
    Parameters
        ----------
        env : gym environment
        K : int
            Total number of episodes to run.
        algorithm : str
            {"UCB_H", "UCBVI", "UCB_f"}.
        arg : dict
            Contain the value of ζ/2.
        ----------
    """
    
    # Get problem cardinalities
    card = {"S": env.observation_space.n, 
            "A": env.action_space.n,
            "H": env.horizon,
            "K": K,
           }
    

    ########################
    ### Global arguments ###
    ########################

    if "p" not in arg: # Probability of failure
        arg["p"] = 0.01
        
    if "C" not in arg: # Scaling of bonus [0, 1]
        arg["C"] = 0.05
    
    arg["bonus"] = Hoeffding_bonus(card["S"], card["A"], card["H"], arg["p"])
    
    arg["alpha"] = lambda t: (card["H"]+1)/(card["H"]+t)
    
    # Initial Q-value
    Q = card["H"]*np.ones((card["H"], card["S"], card["A"]))
    
    # Set Q_H(s,a) = r(s,a) for all algorithms
    Q[-1,:] = np.copy(env.R[-1,:])
    

    #########################################
    ### Initialization for each algorithm ###
    #########################################

    if algorithm == "UCB_H":
        
        # Counter
        N = np.zeros((card["H"], card["S"], card["A"]))
    
    elif algorithm == "UCBVI":
        
        # Counters
        N_1 = np.zeros((card["H"], card["S"], card["A"]))
        N_2 = np.zeros((card["H"], card["S"], card["S"], card["A"]))
        
        # Rewards
        arg["r"] = np.copy(env.R)
        
    elif algorithm == "UCB_f":
        
        # Counter
        N = 0
        
        # Model or noisy model
        if arg["epsilon"] > 0:
            arg["f"] = corrupt_model_with_random_noise(np.copy(env.F), card, arg["epsilon"])
        else:
            arg["f"] = np.copy(env.F)
        
        # Rewards and Lipschitz constant
        arg["r"] = np.copy(env.R)
        arg["L"] = env.L
    
    else:
        
        raise ValueError("Unrecognized algorithm.")
    

    #################
    ### Main loop ###
    #################

    regret = []
    
    for k in range(K):
        
        # Trajectory from greedy policy
        trajectory = trajectory_greedy_policy(env, Q)

        # Evaluate policy at k
        V = policy_evaluation(env, Q)

        # Calculate regret
        regret.append(np.mean(env.V_star[0]) - np.mean(V[0]))
        
        # Q-learning
        if algorithm == "UCB_H":
            
            Q, N = UCB_H(Q, N, card, arg, trajectory)
        
        if algorithm == "UCBVI":
            
            Q, N_1, N_2 = UCBVI(Q, N_1, N_2, card, arg, trajectory)
        
        elif algorithm == "UCB_f":
            
            Q, N = UCB_f(Q, N, card, arg, trajectory)
    
    regret = np.array(regret)
    
    return regret


########################
### Helper functions ###
########################


def trajectory_greedy_policy(env, Q):
    """Generate a trajectory for one episode using the greedy policy w.r.t. Q"""

    state = env.reset()
    trajectory = {"state":[], "action":[], "state_next":[], "reward":[]}
    
    for horizon in range(env.horizon):
        
        # Greedy action
        action = np.argmax(Q[horizon, state])

        # MDP Transition
        state_next, reward, done, _ = env.step(action)
        
        # Store (s,a,s',r)
        trajectory["state"].append(state)
        trajectory["action"].append(action)
        trajectory["state_next"].append(state_next)
        trajectory["reward"].append(reward)
        
        # Update state
        state = state_next
        
    return trajectory


def Hoeffding_bonus(S, A, H, p):
    """Hoeffding-style bonus"""

    log_factor = np.log(S*A*H/p)
    bonus = lambda rng, t: np.sqrt((rng**2)*log_factor/t)
    
    return bonus


def generate_Lipschitz_env(S, A, H, W, L):
    """Sample an MDP with Lipschitz constant L"""

    # Loop until a larger Lipschitz constant than L is satisfied
    # TODO: make this more efficient and able to handle large L
    if L > 1.01: raise ValueError("L is too large, the loop might not converge.")
    while True:
        
        # Sample Environment 
        env = MDP(S, A, H, W)
        
        # Exact Dynamic Programming
        Q_star = value_iteration(env)
        V_star = policy_evaluation(env, Q_star)
        
        # Lipschitz Constant
        c = max(abs(V_star[0][1:] - V_star[0][:-1]))
        
        if c >= L:
            break
    
    # Normalize reward to get the desired Lipschitz constant
    env.R = env.R * (L/c)
    
    # Store Dynamic Programming results
    env.Q_star = value_iteration(env)
    env.V_star = policy_evaluation(env, env.Q_star)
    env.V_greedy = policy_evaluation(env, env.R)
    env.L = L
    
    return env


def corrupt_model_with_random_noise(f, card, epsilon):
    """Corrupt f with random noise supported on [-epsilon, epsilon]"""

    # Sample random noise
    model_noise = np.random.randint(low=-epsilon, high=epsilon+1, size=(card["S"], card["A"]))
    model_noise = np.repeat(model_noise[np.newaxis, :, :], card["H"], axis=0)
    
    # Corrupt_model
    f_noisy = f + model_noise
    f_noisy = np.clip(f_noisy, 0, card["S"]-1)
    
    return f_noisy
