import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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


def print_grid_triangles(values=None, cell_width=12):
    """
    Print a 3x3 grid with each cell divided into 4 triangles showing Q-values.
    
    Args:
        values: Optional dictionary mapping (state, action) -> value
        cell_width: Width of each cell in characters
    """
    # Define the characters for drawing the grid
    horizontal = '─' * cell_width
    vertical = '│'
    corner = '┼'
    top_corner = '┬'
    bottom_corner = '┴'
    left_corner = '├'
    right_corner = '┤'
    top_left = '┌'
    top_right = '┐'
    bottom_left = '└'
    bottom_right = '┘'
    diagonal = '/'
    back_diagonal = '\\'
    
    # Create the grid lines
    def create_line(is_top=False, is_bottom=False):
        line = []
        for i in range(3):
            if i == 0:
                line.append(top_left if is_top else bottom_left if is_bottom else left_corner)
            line.append(horizontal)
            if i < 2:
                line.append(top_corner if is_top else bottom_corner if is_bottom else corner)
        line.append(top_right if is_top else bottom_right if is_bottom else right_corner)
        return ''.join(line)
    
    # Print the grid
    print(create_line(is_top=True))
    for row in range(3):
        # Print the main cell content
        for sub_row in range(3):  # 3 rows to properly show the triangles
            line = []
            for col in range(3):
                state = row * 3 + col + 1
                if values:
                    up_val = f"{values.get((state, 'up'), 0):.2f}"
                    right_val = f"{values.get((state, 'right'), 0):.2f}"
                    left_val = f"{values.get((state, 'left'), 0):.2f}"
                    down_val = f"{values.get((state, 'down'), 0):.2f}"
                    
                    if sub_row == 0:
                        # Top row with up and right triangles
                        content = f"{up_val:^{cell_width//2}}{back_diagonal}{right_val:^{cell_width//2}}"
                    elif sub_row == 1:
                        # Middle row with diagonals
                        content = f"{left_val:^{cell_width//2}}{diagonal}{right_val:^{cell_width//2}}"
                    else:
                        # Bottom row with left and down triangles
                        content = f"{left_val:^{cell_width//2}}{back_diagonal}{down_val:^{cell_width//2}}"
                else:
                    if sub_row == 0:
                        content = f"{state:^{cell_width//2}}{back_diagonal}{' ':^{cell_width//2}}"
                    elif sub_row == 1:
                        content = f"{' ':^{cell_width//2}}{diagonal}{' ':^{cell_width//2}}"
                    else:
                        content = f"{' ':^{cell_width//2}}{back_diagonal}{' ':^{cell_width//2}}"
                
                line.append(vertical + content)
            line.append(vertical)
            print(''.join(line))
        if row < 2:
            print(create_line())
    print(create_line(is_bottom=True))


def plot_values(values=None, horizon=2, policy="optimal", color_scheme="bw"):
    """
    Plot the grid world with either Q-values or V-values using matplotlib.
    
    Args:
        values: Either:
            - Dictionary mapping (state, action) -> value for Q-values
            - Dictionary mapping state -> value for V-values
        horizon: The planning horizon
        policy: String describing the policy ("optimal", "always_up", etc.)
        color_scheme: Color scheme to use ("bw" for black and white, "color" for red/blue)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up the grid
    for i in range(4):
        ax.axhline(y=i, color='black', linewidth=2)
        ax.axvline(x=i, color='black', linewidth=2)
    
    # Set limits and remove ticks
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Check if values are Q-values or V-values
    is_q_values = values and isinstance(next(iter(values)), tuple)
    
    # Define colors for different value ranges
    def get_color(value):
        if color_scheme == "color":
            if value > 0:
                return plt.cm.Reds(min(1, value/10))
            else:
                return plt.cm.Blues(min(1, abs(value)/10))
        else:  # bw
            return 'white'
    
    # Draw cells and add values
    for row in range(3):
        for col in range(3):
            state = row * 3 + col + 1
            x, y = col, 2-row  # Convert to matplotlib coordinates
            
            if values:
                if is_q_values:
                    # Q-values: draw triangles
                    triangles = [
                        # Up triangle
                        [(x, y+1), (x+1, y+1), (x+0.5, y+0.5)],
                        # Right triangle
                        [(x+1, y+1), (x+1, y), (x+0.5, y+0.5)],
                        # Down triangle
                        [(x+1, y), (x, y), (x+0.5, y+0.5)],
                        # Left triangle
                        [(x, y), (x, y+1), (x+0.5, y+0.5)]
                    ]
                    
                    q_values = {
                        'up': values.get((state, 'up'), 0),
                        'right': values.get((state, 'right'), 0),
                        'down': values.get((state, 'down'), 0),
                        'left': values.get((state, 'left'), 0)
                    }
                    
                    for i, (triangle, (action, value)) in enumerate(zip(triangles, q_values.items())):
                        color = get_color(value)
                        polygon = Polygon(triangle, facecolor=color, edgecolor='black', linestyle='--', linewidth=1)
                        ax.add_patch(polygon)
                        
                        center_x = sum(p[0] for p in triangle) / 3
                        center_y = sum(p[1] for p in triangle) / 3
                        ax.text(center_x, center_y, f"{value:.2f}", 
                               ha='center', va='center', fontsize=19)
                else:
                    # V-values: just show the value in the center
                    value = values.get(state, 0)
                    color = get_color(value)
                    rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black')
                    ax.add_patch(rect)
                    ax.text(x+0.5, y+0.5, f"{value:.2f}", 
                           ha='center', va='center', fontsize=19)
            else:
                # Just show state number
                ax.text(x+0.5, y+0.5, str(state), 
                       ha='center', va='center', fontsize=12)
                
                # Draw dashed lines for empty grid
                center = (x+0.5, y+0.5)
                # Draw diagonal lines
                ax.plot([x, x+1], [y+1, y], 'k--', linewidth=1)  # Top-right to bottom-left
                ax.plot([x, x+1], [y, y+1], 'k--', linewidth=1)  # Bottom-right to top-left
    
    # Add title
    value_type = "Q-values" if is_q_values else "V-values"
    ax.set_title(f"Mario World, {value_type}, {policy} policy, horizon {horizon}")
    
    plt.tight_layout()
    plt.show()


def always_up_policy_value(mdp, horizon):
    """
    Compute the H‐step state values V_H(s) for the 'always up' policy.
    This policy always chooses the 'up' action regardless of the state.

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
            # For the 'always up' policy, we only consider the 'up' action
            a = 'up'
            # Compute V_t(s) = R(s,a) + γ * Σ_{s'} P(s,a,s') * V_{t-1}(s')
            v = mdp.R[(s,a)]
            v += mdp.gamma * sum(prob * V[s2] for prob, s2 in mdp.P[(s,a)])
            V_next[s] = v
        V = V_next

    return V


# Example usage
if __name__ == "__main__":
    mdp = GridMDP()
    H = 2
    Q_H = finite_horizon_q_value(mdp, H)
    V_H = finite_horizon_value(mdp, H)
    V_up = always_up_policy_value(mdp, H)
    
    # Plot Q-values (black and white)
    plot_values(Q_H, horizon=H, policy="optimal")
    
    # Plot V-values (black and white)
    plot_values(V_H, horizon=H, policy="optimal")
    
    # Plot always-up policy values (black and white)
    plot_values(V_up, horizon=H, policy="always_up")
    
    # Plot with colors
    plot_values(Q_H, horizon=H, policy="optimal", color_scheme="color")