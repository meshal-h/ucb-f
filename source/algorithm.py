import numpy as np


##################
### Algorithms ###
##################


def UCB_f(Q, N, card, arg, trajectory):
    
    """
    UCB-f: Algorithm 1 in the paper.
    
    Parameters
        ----------
        Q : numpy array [H,S,A]
            The Q-values estimates.
        N : int
            Counter for how many time the Q has been updated.
        card : dict
            Contain the problem cardinalities.
        arg : dict
            Contain
                - arg["p"]: float for the probability of failure.
                - arg["C"]: float for the bonus constant.
                - arg["alpha"]: lambda function for learning rate.
                - arg["bonus"]: lambda function for bonus.
                - arg["f"]: numpy array [H,S,A] for the approximate model.
                - arg["r"]: numpy array [H,S,A] for the reward function.
                - arg["epsilon"]: ζ/2 in the paper.
                - arg["L"]: Lipschitz constant for the optimal value function.
        trajectory: dict
            Contain data for a single trajectory.
        ----------
    """
    
    for h in range(card["H"]):
        
        # Get (s,a,s')
        s      = trajectory["state"][h]
        a      = trajectory["action"][h]
        s_next = trajectory["state_next"][h]
        
        # Range of the value function
        rng = card["H"]-h
        
        # Transition randomness
        w = s_next - (arg["f"][h,s,a])
        
        # Increase counter
        N += 1
        
        # Calculate bonus
        b = arg["C"]*(arg["bonus"](rng, N) + arg["epsilon"]*arg["L"])
        
        for s_par in range(card["S"]):
            
            for a_par in range(card["A"]):
                
                s_par_next = (arg["f"][h,s_par,a_par] + w) % card["S"]
                
                # Calculate target
                if h == card["H"]-1:
                    target = arg["r"][h,s_par,a_par]
                else:
                    target = arg["r"][h,s_par,a_par] + min(card["H"], np.max(Q[h+1,s_par_next])) + b
                
                # Q-update
                alpha = arg["alpha"](N)
                Q[h,s_par,a_par] = (1-alpha)*Q[h,s_par,a_par] + alpha*target
    
    return Q, N


def UCB_H(Q, N, card, arg, trajectory):
    
    """
    UCB-H (https://arxiv.org/abs/1807.03765)
    
    Parameters
        ----------
        Q : numpy array [H,S,A]
            Q-values estimates.
        N : numpy array [H,S,A]
            Counter for how many time the Q has been updated.
        card : dict
            Contain the problem cardinalities.
        arg : dict
            Contain
                - arg["p"]: float for the probability of failure.
                - arg["C"]: float for the bonus constant.
                - arg["alpha"]: lambda function for learning rate.
                - arg["bonus"]: lambda function for bonus.
        trajectory: dict
            Contain data for a single trajectory.
        ----------
    """
    
    for h in range(card["H"]):
        
        # Get (s,a,s',r)
        s      = trajectory["state"][h]
        a      = trajectory["action"][h]
        s_next = trajectory["state_next"][h]
        r      = trajectory["reward"][h]
        
        # Range of the value function
        rng = card["H"]-h
        
        # Increase counter
        N[h,s,a] += 1
        
        # Calculate bonus
        b = arg["C"]*arg["bonus"](rng, N[h,s,a])
        
        # Calculate target
        if h == card["H"]-1:
            target = r
        else:
            target = r + min(card["H"], np.max(Q[h+1,s_next])) + b
                
        # Q-update
        alpha = arg["alpha"](N[h,s,a])
        Q[h,s,a] = (1-alpha)*Q[h,s,a] + alpha*target
    
    return Q, N


def UCBVI(Q, N_1, N_2, card, arg, trajectory):
    
    """
    UCBVI (https://arxiv.org/abs/1703.05449)
    
    Parameters
        ----------
        Q : numpy array [H,S,A]
            Q-values estimates.
        N_1 : numpy array [H,S,A]
            Counter for state transitions (h,s,a).
        N_2 : numpy array [H,S,S,A]
            Counter for state transitions (h,s,s_next,a).
        card : dict
            Contain the problem cardinalities.
        arg : dict
            Contain
                - arg["p"]: float for the probability of failure.
                - arg["C"]: float for the bonus constant.
                - arg["bonus"]: lambda function for bonus.
                - arg["r"]: numpy array [H,S,A] for the reward function.
        trajectory: dict
            Contain data for a single trajectory.
        ----------
    """
    
    # Update transition probability estimates from trajectory data
    for h in range(card["H"]):
        
        # Get (s,a,s')
        s      = trajectory["state"][h]
        a      = trajectory["action"][h]
        s_next = trajectory["state_next"][h]
        
        # Increase counters
        N_1[h,s,a] += 1
        N_2[h,s,s_next,a] += 1
    
    # Value iteration
    for h in range(card["H"]-1):
        
        for s in range(card["S"]):
            
            for a in range(card["A"]):
                
                if N_1[h,s,a] >= 1:
                    
                    # Range of the value function
                    rng = card["H"]-h

                    # Calculate bonus
                    b = arg["C"]*arg["bonus"](rng, N_1[h,s,a])

                    # Empirical value function
                    V_hat = arg["r"][h,s,a] + np.dot( (N_2[h,s,:,a]/ N_1[h,s,a]), np.max(Q[h+1], axis=1) ) + b
                    
                    # Q-update
                    Q[h,s,a] = min(card["H"], V_hat)
                    
                else:
                    Q[h,s,a] = card["H"]
    
    return Q, N_1, N_2
