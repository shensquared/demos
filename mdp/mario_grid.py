import numpy as np

class GridMDP:
    def __init__(self):
        # States and actions
        self.states = list(range(1,10))               # 1 through 9
        self.actions = ['up','down','left','right']   # action space
        self.gamma = 0.9                              # discount factor
        
        # Transition and reward tables
        # P[(s,a)] = [(probability, next_state), ...]
        # R[(s,a)] = immediate reward for taking a in s
        self.P = {(s,a): [] for s in self.states for a in self.actions}
        self.R = {(s,a): 0  for s in self.states for a in self.actions}
        
        # Helpers: state ↔ (row,col)
        # Create a 3x3 grid mapping
        self._s2rc = {}
        self._rc2s = {}
        for s in self.states:
            row = (s-1) // 3
            col = (s-1) % 3
            self._s2rc[s] = (row, col)
            self._rc2s[(row, col)] = s
        
        # Build tables
        for s in self.states:
            for a in self.actions:
                # special stochastic in state 6, action 'up'
                if s == 6 and a == 'up':
                    self.P[(s,a)] = [(0.2, 2), (0.8, 3)]
                else:
                    r, c = self._s2rc[s]
                    if   a == 'up':    r2, c2 = max(r-1,0),   c
                    elif a == 'down':  r2, c2 = min(r+1,2),   c
                    elif a == 'left':  r2, c2 =   r,     max(c-1,0)
                    elif a == 'right': r2, c2 =   r,     min(c+1,2)
                    
                    # If we hit a wall, stay in the same state
                    if (r2, c2) not in self._rc2s:
                        s2 = s
                    else:
                        s2 = self._rc2s[(r2,c2)]
                    self.P[(s,a)] = [(1.0, s2)]
                
                # rewards
                if s == 3:
                    self.R[(s,a)] =  1
                elif s == 6:
                    self.R[(s,a)] = -10
                else:
                    self.R[(s,a)] =   0

    def print_transitions(self):
        """Prints for every state and action all (next_state,probability,reward)."""
        for s in self.states:
            for a in self.actions:
                entries = self.P[(s,a)]
                r = self.R[(s,a)]
                for prob, s2 in entries:
                    print(f"s={s:>2}, a={a:<5} → s'={s2}  (prob={prob:.1f}, reward={r:+d})")


def finite_horizon_value(mdp, horizon):
    """
    Compute the H‐step optimal state values V_H(s) for a finite‐horizon MDP.

    Args:
        mdp: an object with
            - mdp.states: iterable of states
            - mdp.actions: iterable of actions
            - mdp.P[(s,a)]: list of (probability, next_state) tuples
            - mdp.R[(s,a)]: immediate reward for (s,a)
            - mdp.gamma: discount factor
        horizon: int, number of steps H

    Returns:
        dict mapping state -> V_H(state)
    """
    # V_0(s) = 0 for all s
    V = {s: 0.0 for s in mdp.states}

    # iterate t = 1…H
    for t in range(1, horizon + 1):
        V_next = {}
        for s in mdp.states:
            # for each action, compute Q_t(s,a) = R(s,a) + γ * Σ_{s'} P(s,a,s') * V_{t-1}(s')
            q_values = []
            for a in mdp.actions:
                q = mdp.R[(s,a)]
                q += mdp.gamma * sum(prob * V[s2] for prob, s2 in mdp.P[(s,a)])
                q_values.append(q)
            # optimal value is max over actions
            V_next[s] = max(q_values)
        V = V_next

    return V


def finite_horizon_q_value(mdp, horizon):
    """
    Compute the H‐step optimal Q-values Q_H(s,a) for a finite‐horizon MDP.

    Args:
        mdp: an object with
            - mdp.states: iterable of states
            - mdp.actions: iterable of actions
            - mdp.P[(s,a)]: list of (probability, next_state) tuples
            - mdp.R[(s,a)]: immediate reward for (s,a)
            - mdp.gamma: discount factor
        horizon: int, number of steps H

    Returns:
        dict mapping (state, action) -> Q_H(state, action)
    """
    # Q_0(s,a) = 0 for all s,a
    Q = {(s,a): 0.0 for s in mdp.states for a in mdp.actions}

    # iterate t = 1…H
    for t in range(1, horizon + 1):
        Q_next = {}
        for s in mdp.states:
            for a in mdp.actions:
                # Q_t(s,a) = R(s,a) + γ * Σ_{s'} P(s,a,s') * max_{a'} Q_{t-1}(s',a')
                q = mdp.R[(s,a)]
                q += mdp.gamma * sum(
                    prob * max(Q[(s2,a2)] for a2 in mdp.actions)
                    for prob, s2 in mdp.P[(s,a)]
                )
                Q_next[(s,a)] = q
        Q = Q_next

    return Q


# Example usage
if __name__ == "__main__":
    mdp = GridMDP()
    print(mdp.print_transitions())
    H = 2# e.g., 5‐step horizon
    V_H = finite_horizon_value(mdp, H)
    Q_H = finite_horizon_q_value(mdp, H)
    print(f"Optimal {H}-step state values:")
    for s in sorted(V_H):
        print(f"  V_{H}({s}) = {V_H[s]:.4f}")
    print(f"\nOptimal {H}-step Q-values:")
    for s in sorted(mdp.states):
        for a in mdp.actions:
            print(f"  Q_{H}({s},{a}) = {Q_H[(s,a)]:.4f}")