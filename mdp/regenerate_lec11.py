"""
Regenerate all lec11 grid images with consistent Palatino font.
Maps each hash-named PNG to its Q-value content and regenerates it.
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mario_grid import plot_custom_grid

SLIDE_DIR = "/Users/shenshen/code/slides/390/spring26-slides/slides_introml-sp26-lec11/introml-sp26-lec11"
QLEARN_DIR = "/Users/shenshen/code/demos/mdp/qlearning_steps"
OUT_DIR = "/Users/shenshen/code/demos/mdp/lec11_regen"

os.makedirs(OUT_DIR, exist_ok=True)

states = list(range(1, 10))
actions = ['up', 'down', 'left', 'right']

def all_zeros():
    return {(s, a): 0.0 for s in states for a in actions}

def full_q(overrides):
    """All zeros with specific overrides."""
    q = all_zeros()
    q.update(overrides)
    return q

# ============================================================
# Define all grid image mappings
# ============================================================

grid_images = {
    # 1. Empty grid (no numbers)
    "1aceec64e14a3d838da49d94219be330": ("empty", {}),

    # 2. Sparse: only Q(1,up)=0
    "9be1ca7abcd471ccc996be3de704f4ed": ("sparse", {(1, 'up'): 0}),

    # 3. Sparse: Q(1,up)=0, Q(1,down)=0
    "847fb1bd01d887d5297ac459ef57b0b3": ("sparse", {(1, 'up'): 0, (1, 'down'): 0}),

    # 4. All zeros (full grid)
    "6176510bc3f76252046bd24c9daef9e7": ("full", all_zeros()),

    # 5. All zeros (full grid, duplicate)
    "70eef90f77e8c48c2cedce0c947dd25c": ("full", all_zeros()),

    # 6. Sparse: s1 all=0, s2 all=0, s3 up=1 only
    "e20e0b52dbdfc2ca447590add1362f71": ("sparse", {
        (1, 'up'): 0, (1, 'down'): 0, (1, 'left'): 0, (1, 'right'): 0,
        (2, 'up'): 0, (2, 'down'): 0, (2, 'left'): 0, (2, 'right'): 0,
        (3, 'up'): 1,
    }),

    # 7. Sparse: s1 all=0, s2 all=0, s3 up=1, down=1
    "f02df040444d9301cfe485b7269cbde4": ("sparse", {
        (1, 'up'): 0, (1, 'down'): 0, (1, 'left'): 0, (1, 'right'): 0,
        (2, 'up'): 0, (2, 'down'): 0, (2, 'left'): 0, (2, 'right'): 0,
        (3, 'up'): 1, (3, 'down'): 1,
    }),

    # 8. Sparse: s1-2 all=0, s3 all=1, s4-5 all=0, s6-9 empty
    "7083bf00e87245b30d28b6ffb70ab84f": ("sparse", {
        (1, 'up'): 0, (1, 'down'): 0, (1, 'left'): 0, (1, 'right'): 0,
        (2, 'up'): 0, (2, 'down'): 0, (2, 'left'): 0, (2, 'right'): 0,
        (3, 'up'): 1, (3, 'down'): 1, (3, 'left'): 1, (3, 'right'): 1,
        (4, 'up'): 0, (4, 'down'): 0, (4, 'left'): 0, (4, 'right'): 0,
        (5, 'up'): 0, (5, 'down'): 0, (5, 'left'): 0, (5, 'right'): 0,
    }),

    # 9. Sparse: s1-2 all=0, s3 all=1, s4-5 all=0, s6 up=-10, s7-9 empty
    "dea2bcb13f0a269b1336e902f6d9d29c": ("sparse", {
        (1, 'up'): 0, (1, 'down'): 0, (1, 'left'): 0, (1, 'right'): 0,
        (2, 'up'): 0, (2, 'down'): 0, (2, 'left'): 0, (2, 'right'): 0,
        (3, 'up'): 1, (3, 'down'): 1, (3, 'left'): 1, (3, 'right'): 1,
        (4, 'up'): 0, (4, 'down'): 0, (4, 'left'): 0, (4, 'right'): 0,
        (5, 'up'): 0, (5, 'down'): 0, (5, 'left'): 0, (5, 'right'): 0,
        (6, 'up'): -10,
    }),

    # 10-14. Q-learning steps 5-9 (full grids, copy from qlearning_steps)
    # NOTE: dropped (3,down,1,6); these are the actual hashes the slides reference
    "1e06f8dd209f9c72c7f76fbe0bf55e28": ("qlearn_step", 5),  # Q(3,up)=0.7
    "4576260dba2284d9868060e086556b82": ("qlearn_step", 6),  # +Q(6,up)=-6.56
    "dde7e50ecfc42711b781ede85dab58ac": ("qlearn_step", 7),  # Q(6,up)=-8.97
    "cc4ec9951ef9a72814f126a3de98714f": ("qlearn_step", 8),  # Q(6,up)=-9.25
    "711bdba158de2e1bc3179045ba60a5cd": ("qlearn_step", 9),  # Q(6,up)=-9.77

    # 16-17. Q*_1: full grid, Q(3,all)=1, Q(6,all)=-10, rest=0
    "7d9156bc5ec60e36c595cbd345482c8e": ("full", full_q({
        (3, 'up'): 1, (3, 'down'): 1, (3, 'left'): 1, (3, 'right'): 1,
        (6, 'up'): -10, (6, 'down'): -10, (6, 'left'): -10, (6, 'right'): -10,
    })),
    "801c078c4abaa57a103e9461ea674a19": ("full", full_q({
        (3, 'up'): 1, (3, 'down'): 1, (3, 'left'): 1, (3, 'right'): 1,
        (6, 'up'): -10, (6, 'down'): -10, (6, 'left'): -10, (6, 'right'): -10,
    })),

    # 18. Modified Q*_1: Q(6,up)=-9.1 instead of -10
    "d92a600903da7c92119006cfde6e420c": ("full", full_q({
        (3, 'up'): 1, (3, 'down'): 1, (3, 'left'): 1, (3, 'right'): 1,
        (6, 'up'): -9.1, (6, 'down'): -10, (6, 'left'): -10, (6, 'right'): -10,
    })),
}

# ============================================================
# Generate images
# ============================================================

for hash_name, (img_type, data) in grid_images.items():
    out_path = os.path.join(OUT_DIR, f"{hash_name}.png")

    if img_type == "empty":
        # Empty grid - pass empty dict
        plot_custom_grid({}, save_path=out_path)
        print(f"  Generated empty grid -> {hash_name[:8]}")

    elif img_type == "sparse":
        # Sparse grid - pass partial dict (only specified cells get numbers)
        plot_custom_grid(data, save_path=out_path)
        print(f"  Generated sparse grid -> {hash_name[:8]}")

    elif img_type == "full":
        # Full grid - all 36 cells shown
        plot_custom_grid(data, save_path=out_path)
        print(f"  Generated full grid -> {hash_name[:8]}")

    elif img_type == "qlearn_step":
        # Copy from pre-generated Q-learning steps
        step_num = data
        src = os.path.join(QLEARN_DIR, f"step_{step_num:04d}.png")
        if os.path.exists(src):
            shutil.copy2(src, out_path)
            print(f"  Copied Q-learn step {step_num} -> {hash_name[:8]}")
        else:
            print(f"  ERROR: Missing {src}")

print(f"\nGenerated {len(grid_images)} images in {OUT_DIR}")

# ============================================================
# Copy to slide directory
# ============================================================

print(f"\nCopying to {SLIDE_DIR}...")
for hash_name in grid_images:
    src = os.path.join(OUT_DIR, f"{hash_name}.png")
    dst = os.path.join(SLIDE_DIR, f"{hash_name}.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  {hash_name[:8]} -> slides")
    else:
        print(f"  ERROR: Missing {src}")

print("\nDone!")
