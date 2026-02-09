import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # When the Closed-Form Solution Fails

    The closed-form solution $\theta^* = (X^\top X)^{-1} X^\top Y$ requires $X^\top X$ to be invertible.

    Let's see what happens when it's not...
    """)
    return


@app.cell
def _():
    import numpy as np
    np.set_printoptions(precision=3)

    def to_latex(arr, name=""):
        """Convert numpy array to LaTeX bmatrix"""
        if arr.ndim == 1:
            rows = " \\\\ ".join(f"{x:.2f}" for x in arr)
            matrix = f"\\begin{{bmatrix}} {rows} \\end{{bmatrix}}"
        else:
            rows = " \\\\ ".join(" & ".join(f"{x:.2f}" for x in row) for row in arr)
            matrix = f"\\begin{{bmatrix}} {rows} \\end{{bmatrix}}"
        if name:
            return f"{name} = {matrix}"
        return matrix

    return np, to_latex


@app.cell
def _(mo):
    mo.md(r"""
    ## Case (a): $n < d$ — More features than data points
    """)
    return


@app.cell
def _(mo, np, to_latex):
    # 1 data point, 2 features
    X_a = np.array([[2, 3]])
    y_a = np.array([4])

    mo.md(rf"""
    **Data:** 1 sample, 2 features $(n=1, d=2)$

    ${to_latex(X_a, "X")}$ $\quad$ ${to_latex(y_a, "Y")}$
    """)
    return X_a, y_a


@app.cell
def _(X_a, mo, np, to_latex):
    # X^T X is 2x2 but rank 1
    XtX_a = X_a.T @ X_a

    mo.md(rf"""
    ${to_latex(XtX_a, "X^\\top X")}$

    $\text{{rank}}(X^\top X) = {np.linalg.matrix_rank(XtX_a)}$ $\quad$ $\det(X^\top X) = {np.linalg.det(XtX_a):.4f}$
    """)
    return (XtX_a,)


@app.cell
def _(XtX_a, mo, np):
    # Try to invert — this will fail!
    try:
        np.linalg.inv(XtX_a)
        msg = "✅ Invertible"
    except np.linalg.LinAlgError as e:
        msg = f"❌ **LinAlgError:** {e}"

    mo.md(msg)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Case (b): Collinear Features
    """)
    return


@app.cell
def _(mo, np, to_latex):
    # 3 data points, but features are collinear (x2 = 1.5 * x1)
    X_b = np.array([[2, 3],
                    [4, 6],
                    [6, 9]])
    y_b = np.array([4, 8, 12])

    mo.md(rf"""
    **Data:** 3 samples, 2 features $(n=3, d=2)$, but $x_2 = 1.5 \cdot x_1$

    ${to_latex(X_b, "X")}$ $\quad$ ${to_latex(y_b, "Y")}$
    """)
    return X_b, y_b


@app.cell
def _(X_b, mo, np, to_latex):
    # X^T X is 2x2 but still rank 1
    XtX_b = X_b.T @ X_b

    mo.md(rf"""
    ${to_latex(XtX_b, "X^\\top X")}$

    $\text{{rank}}(X^\top X) = {np.linalg.matrix_rank(XtX_b)}$ $\quad$ $\det(X^\top X) = {np.linalg.det(XtX_b):.4f}$
    """)
    return (XtX_b,)


@app.cell
def _(XtX_b, mo, np):
    # Try to invert — this will also fail!
    try:
        np.linalg.inv(XtX_b)
        msg = "✅ Invertible"
    except np.linalg.LinAlgError as e:
        msg = f"❌ **LinAlgError:** {e}"

    mo.md(msg)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Fix: Regularization (Preview)

    Ridge regression adds $\lambda I$ to make it invertible:
    """)
    return


@app.cell
def _(mo):
    lambda_slider = mo.ui.slider(0.01, 2.0, value=0.1, step=0.01, label="λ")
    lambda_slider
    return (lambda_slider,)


@app.cell
def _(X_b, lambda_slider, mo, np, to_latex, y_b):
    # With regularization, it works!
    lam = lambda_slider.value
    XtX_reg = X_b.T @ X_b + lam * np.eye(2)
    theta = np.linalg.inv(XtX_reg) @ X_b.T @ y_b

    mo.md(rf"""
    $\lambda = {lam:.2f}$

    ${to_latex(XtX_reg, "X^\\top X + \\lambda I")}$

    $\det(X^\top X + \lambda I) = {np.linalg.det(XtX_reg):.4f}$ ✅ **Now invertible!**

    ${to_latex(theta, "\\theta")}$
    """)
    return


if __name__ == "__main__":
    app.run()
