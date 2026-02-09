import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("# Hello from WASM!")
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10, value=5, label="Pick a number")
    slider  # This displays the slider
    return (slider,)


@app.cell
def _(slider, mo):
    result = mo.md(f"""
    You picked: **{slider.value}**

    Squared: **{slider.value ** 2}**
    """)
    result  # This displays the result
    return


if __name__ == "__main__":
    app.run()
