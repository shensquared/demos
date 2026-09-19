"""Choosing lambda by hold-out validation, and by 5-fold cross-validation.

    demos/mdp/venv/bin/python ridge/validation_demo.py

Twelve figures, in two sets of six. Both sets use the same ten points and the same
three lambdas, so they can be shown one after the other.

Hold-out, one split. Each image is a single plot, authored for a 400px-wide block:

    validation-train-lam{0p1,1,10}.png    the 8 training points and the ridge line
    validation-val-lam{0p1,1,10}.png      the 2 held-out points, the line, and E_val

Cross-validation, five splits. Each image is a 3x2 grid authored for a 1200px-wide
block. Lambda sits in the top-left cell and the five folds fill the rest, two across the
top and three along the bottom. On the validation grids that cell also carries the
average of the five fold errors, which is the single number lambda is judged by. Each fold is named inside its own panel rather than
above it, which buys back the row of height a title would take, and the upper left of
every panel is empty because the data rises left to right. Each cell ends up 400x300,
the same size as one of the hold-out images above, so the panels match:

    validation-cv-train-lam{0p1,1,10}.png  each fold's 8 training points and its line
    validation-cv-val-lam{0p1,1,10}.png    each fold's 2 held-out points, and E_i

Sizing is the reason for the two widths, not taste. Type has to clear the floors in
slides/10-250.md, 19 deck px for anything and 29 for anything a student must read. A
figure authored at its display width puts a point at 1.39 deck px, so 21pt titles land
at 29 and 14pt ticks at 19. A single plot does that comfortably inside 400px. Five
subplots inside 400px would give each about 80px of width, which no type size rescues,
so the cross-validation grids are authored at 1200 instead.

Folds are {i, i+5}, so each spans the x range rather than taking a contiguous block of
it, and 10 points in 5 folds holds out 2 at a time. Fold 3 is the hold-out split, so the
first six images are one fold of the second six rather than a separate example.

What these show and do not show. They show the procedure: hold data out, measure error
on it, and with five folds, average the five numbers. They are not an overfitting
demonstration, and a straight line through eight points cannot overfit. One consequence
is worth knowing before extending them. Averaged over all five folds this data slightly
prefers lambda 0.1 to lambda 1, 1.52 against 1.72, because there is little here to
regularise. Nothing in these figures shows that, since each cross-validation image fits
a single lambda, but a figure comparing cross-validation error across lambda would.

The data is written out rather than seeded, because a seeded draw would silently redraw
every figure if numpy's generator stream ever changed underneath it.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['lines.scale_dashes'] = False
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# The ridge demos' palette, so these sit beside them without clashing.
INK, MUTED, GRID = '#2b2b2b', '#5f6672', '#dcdcdc'
DATA, DATA_EDGE, HELD = '#3c78d8', '#1a4a8a', '#674ea7'

X = np.array([0.947, 0.961, 1.224, 2.918, 4.463, 5.120, 5.968, 7.636, 8.077, 8.640])
Y = np.array([0.557, 0.597, 3.422, 2.699, 7.615, 6.664, 8.860, 10.345, 10.132, 11.888])

LAMBDAS = [(0.1, '0p1'), (1.0, '1'), (10.0, '10')]
FOLDS = [np.array([i, i + 5]) for i in range(5)]
HOLDOUT = 2                            # fold 3, holding out x = 1.22 and 7.64

XLIM, YLIM = (0.4, 9.2), (-1.0, 14.5)
# A point is 1.39 deck px whenever a figure is authored at the width it is displayed at,
# which both sets below are. 21pt puts the titles at 29.5, just over the 29 floor for
# text a student must read. Ticks are 15 rather than 14 because the wider subplot
# spacing in the strips costs a little: at 14 they measured 18.8, under the 19 floor.
TITLE, TICK = 21, 15


def ridge_fit(x, y, lam):
    """Minimise (1/n) sum (theta x + theta0 - y)^2 + lam theta^2.

    The offset is not penalised, which is the usual convention and the one the slides
    use: shrinking theta0 would drag the line toward y = 0 for no good reason. Setting
    both derivatives to zero then gives the closed form below, where the penalty adds
    n*lam to the denominator and so shrinks the slope toward flat.
    """
    n = len(x)
    xb, yb = x.mean(), y.mean()
    sxx = ((x - xb) ** 2).sum()
    sxy = ((x - xb) * (y - yb)).sum()
    theta = sxy / (sxx + n * lam)
    return theta, yb - theta * xb


def mse(x, y, theta, theta0):
    return float(np.mean((theta * x + theta0 - y) ** 2))


def rest_of(fold):
    return np.array([i for i in range(len(X)) if i not in fold])


def style(ax, yticks=True):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=TICK)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_yticks([0, 5, 10])
    if not yticks:
        ax.set_yticklabels([])


def draw(ax, theta, theta0, fold, show_train):
    """One panel: the fitted line, plus either the training points or the held-out pair."""
    xs = np.array(XLIM)
    ax.plot(xs, theta * xs + theta0, '-', color=INK, linewidth=2.2, zorder=3)
    rest = rest_of(fold)
    if show_train:
        ax.plot(X[rest], Y[rest], 'o', markersize=8, color=DATA,
                markeredgecolor=DATA_EDGE, markeredgewidth=1.2,
                linestyle='none', zorder=5)
    else:
        for px, py in zip(X[fold], Y[fold]):
            ax.plot([px, px], [py, theta * px + theta0], '--', color=INK,
                    dashes=(4, 3), linewidth=1.8, zorder=4)
        ax.plot(X[fold], Y[fold], 'o', markersize=10, color=HELD,
                markeredgecolor=INK, markeredgewidth=1.3,
                linestyle='none', zorder=5)


def inline_row(ax, y, pieces):
    """Lay out mixed-size math as one line, positioned from measured widths.

    mathtext has no \\textstyle, so an inline-sized sum cannot be asked for inside a
    single string. Measured at 22pt, \\sum_i sets 57px tall against 32px for the
    symbols either side of it, which is what makes it tower. Drawing that one piece as
    its own artist at 14pt brings it to 37px and puts it back on the line, and mixing
    sizes in turn means placing each piece by hand rather than letting one string lay
    itself out. Each piece is measured off-axes first, then discarded.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    widths = []
    for text, size, _ in pieces:
        probe = ax.text(0, -1, text, fontsize=size)
        fig.canvas.draw()
        widths.append(probe.get_window_extent(r).width)
        probe.remove()
    box = ax.get_window_extent(r)
    x = box.x0 + (box.width - sum(widths)) / 2.0
    to_axes = ax.transAxes.inverted()
    for (text, size, colour), width in zip(pieces, widths):
        ax.text(to_axes.transform((x, box.y0))[0], y, text, fontsize=size,
                color=colour, ha='left', va='center', transform=ax.transAxes)
        x += width


def save(fig, name, block_w, probe=None):
    """Report the type size at the width this image is actually shown at.

    probe is the artist standing in for the panel's name, since the grids label their
    panels from the inside and so have no title to measure.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    scale = block_w / (fig.get_size_inches()[0] * fig.dpi)
    # the first axes carrying ticks: in the grids the lambda cell has none
    ax0 = next(a for a in fig.axes if [t for t in a.get_xticklabels() if t.get_text()])
    label = probe if probe is not None else ax0.title
    label_h = label.get_window_extent(r).height * scale
    tick_h = [t for t in ax0.get_xticklabels()
              if t.get_text()][0].get_window_extent(r).height * scale
    fig.savefig(os.path.join(HERE, name), dpi=fig.dpi, facecolor='white')
    plt.close(fig)
    print('  %-34s label %4.1f  tick %4.1f deck px' % (name, label_h, tick_h))


# ---- six single plots, for 400px blocks -------------------------------------------
fold = FOLDS[HOLDOUT]
print('hold-out: fold %d, x = %s' % (
    HOLDOUT + 1, ', '.join('%.2f' % v for v in X[fold])))
for lam, tag in LAMBDAS:
    th, t0 = ridge_fit(X[rest_of(fold)], Y[rest_of(fold)], lam)
    e = mse(X[fold], Y[fold], th, t0)
    print('  lambda %-5g theta %.3f  train MSE %.2f  E_val %.2f'
          % (lam, th, mse(X[rest_of(fold)], Y[rest_of(fold)], th, t0), e))
    for show_train, name, title in (
            (True, 'validation-train-lam%s.png' % tag, r'$\lambda = %g$' % lam),
            (False, 'validation-val-lam%s.png' % tag,
             r'$\mathcal{E}_{\mathrm{val}} = %.2f$' % e)):
        fig = plt.figure(figsize=(4, 3), dpi=200)      # 800x600, shown at 400x300
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.17, right=0.96, top=0.84, bottom=0.18)
        draw(ax, th, t0, fold, show_train)
        style(ax)
        ax.set_title(title, fontsize=TITLE, color=INK, pad=7)
        save(fig, name, 400)

# ---- six 3x2 grids, for 1200px blocks ---------------------------------------------
print('\n5-fold, one 3x2 grid per lambda, lambda in the top-left cell:')
for lam, tag in LAMBDAS:
    errs = []
    for show_train in (True, False):
        fig = plt.figure(figsize=(12.0, 6.0), dpi=100)   # 1200x600, shown 1:1
        axes = fig.subplots(2, 3)
        fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.09,
                            wspace=0.20, hspace=0.22)
        # Top-left cell carries lambda, and on the validation grids the averaging that
        # collapses the five fold errors into the one number lambda is judged by. The
        # training grids have no errors to average, so lambda sits alone there.
        # Spelled out with its five numbers the sum needs about 420px at this type
        # size and the cell is 400 wide, so it stays symbolic with the result beneath.
        corner = axes[0][0]
        corner.axis('off')
        corner.text(0.5, 0.5 if show_train else 0.62, r'$\lambda = %g$' % lam,
                    fontsize=40, color=INK, ha='center', va='center',
                    transform=corner.transAxes)
        if not show_train:
            # One line. The sum is its own smaller artist so it sits inline rather
            # than towering, and the value keeps its colour; see inline_row.
            inline_row(corner, 0.30, [
                (r'$\mathcal{E}_{\mathrm{val}} = \frac{1}{5}$', 22, INK),
                (r'$\sum_i$', 14, INK),
                (r'$\,\mathcal{E}_i =$', 22, INK),
                (r'$\ %.2f$' % np.mean(errs), 22, HELD),
            ])
        cells = [axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2]]
        probe = None
        for k, f in enumerate(FOLDS):
            th, t0 = ridge_fit(X[rest_of(f)], Y[rest_of(f)], lam)
            e = mse(X[f], Y[f], th, t0)
            if show_train:
                errs.append(e)
            ax = cells[k]
            draw(ax, th, t0, f, show_train)
            style(ax)
            name = (r'fold %d' % (k + 1) if show_train
                    else r'fold %d:  $\mathcal{E}_{%d} = %.2f$' % (k + 1, k + 1, e))
            t = ax.text(0.04, 0.95, name, fontsize=TITLE, color=INK,
                        ha='left', va='top', transform=ax.transAxes)
            probe = probe or t
        save(fig, 'validation-cv-%s-lam%s.png' % ('train' if show_train else 'val', tag),
             1200, probe=probe)
    print('  lambda %-5g folds %s  mean %.2f'
          % (lam, '  '.join('%.2f' % e for e in errs), np.mean(errs)))
