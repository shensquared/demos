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
        self._s2rc = {s: ((s-1)//3, (s-1)%3) for s in self.states}
        self._rc2s = {rc: s for s,rc in self._s2rc.items()}
        
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


# Example usage
if __name__ == "__main__":
    mdp = GridMDP()
    mdp.print_transitions()