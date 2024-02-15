import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np


# Define the convex function and a non-convex function
def convex_function(x):
    return x**2


def non_convex_function(x):
    return np.sin(x)


def function_2d_input(x, y):
    return x**2 + y**2


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div(
    [
        html.P("Select Function:"),
        dcc.Dropdown(
            id="function-dropdown",
            options=[
                {"label": "Convex Function: f(x) = x^2", "value": "convex"},
                {
                    "label": "Non-Convex Function: g(x) = sin(x)",
                    "value": "non-convex",
                },
            ],
            value="convex",  # Default value
        ),
        dcc.Graph(id="function-plot"),
        html.P("Point 1 X-Coordinate:"),
        dcc.Slider(id="point1-x-slider", min=-3, max=3, value=-2, step=1),
        html.P("Point 2 X-Coordinate:"),
        dcc.Slider(id="point2-x-slider", min=-3, max=3, value=1, step=1),
    ]
)


# Callback to update the graph based on inputs
@app.callback(
    Output("function-plot", "figure"),
    [
        Input("function-dropdown", "value"),
        Input("point1-x-slider", "value"),
        Input("point2-x-slider", "value"),
    ],
)
def update_figure(selected_function, point1_x, point2_x):
    # Generate x values
    x = np.linspace(-10, 10, 400)

    # Select the function and update legend label and title based on dropdown value
    if selected_function == "convex":
        y = convex_function(x)
        function_label = "f(x) = x^2"
        title_text = "Demo for Convex Function: f(x) = x^2"
    else:
        y = non_convex_function(x)
        function_label = "g(x) = sin(x)"
        title_text = "Demo for Non-Convex Function: g(x) = sin(x)"

    # Calculate y values for the points based on the selected function
    if selected_function == "convex":
        point1_y = convex_function(point1_x)
        point2_y = convex_function(point2_x)
    else:
        point1_y = non_convex_function(point1_x)
        point2_y = non_convex_function(point2_x)

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=function_label))
    fig.add_trace(
        go.Scatter(
            x=[point1_x, point2_x],
            y=[point1_y, point2_y],
            mode="markers+lines",
            marker=dict(size=10),
            name="Points and Line Segment",
        )
    )

    # Update layout with dynamic title
    fig.update_layout(title=title_text, xaxis_title="x", yaxis_title="y")

    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
