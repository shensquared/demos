import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

# Define The Dataset
cambridge_temps_rough = np.array([78, 79, 81, 82])
cambridge_temps_precise = np.array([78.001, 78.998, 81.002, 82.002])

springfield_temps_rough = np.array([61, 81, 75, 90])
springfield_temps_precise = np.array([60.994, 80.996, 75.004, 90.006])

boston_temps_rough = np.array([76, 78, 82, 84])
boston_temps_precise = np.array([75.994, 77.997, 82.009, 84.006])

worcester_temps = np.array([78, 77, 80, 83])

# Create data points
cambridge_springfield_points_rough = np.column_stack((cambridge_temps_rough, springfield_temps_rough, worcester_temps))
cambridge_springfield_points_precise = np.column_stack((cambridge_temps_precise, springfield_temps_precise, worcester_temps))
cambridge_boston_points_rough = np.column_stack((cambridge_temps_rough, boston_temps_rough, worcester_temps))
cambridge_boston_points_precise = np.column_stack((cambridge_temps_precise, boston_temps_precise, worcester_temps))

# Plotting Utils
def plot_3d_regression(points, title, cols):
    x1 = points[:, 0]
    x2 = points[:, 1]
    y  = points[:, 2]

    X = np.column_stack((x1, x2))
    reg = LinearRegression().fit(X, y)
    a, b = reg.coef_
    c = reg.intercept_

    x1_range = np.linspace(x1.min(), x1.max(), 50)
    x2_range = np.linspace(x2.min(), x2.max(), 50)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    yy = a * xx1 + b * xx2 + c

    scatter = go.Scatter3d(
        x=x1, y=x2, z=y,
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Data Points'
    )

    surface = go.Surface(
        x=xx1, y=xx2, z=yy,
        opacity=0.5,
        colorscale='Viridis',
        name='Regression Plane',
        showscale=False
    )

    annotation_text = f"<b>y = {a:.3f}x₁ + {b:.3f}x₂ + {c:.3f}</b>"
    annotation = dict(
        showarrow=False,
        text=annotation_text,
        xref="paper", yref="paper",
        x=1.0, y=1.0,
        xanchor="right", yanchor="top",
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    layout = go.Layout(
        title=title if title else "Title",
        scene=dict(
            xaxis_title=cols[0] + " temp (x₁)",
            yaxis_title=cols[1] + " temp (x₂)",
            zaxis_title= "Worcester temp (y)",
        ),
        annotations=[annotation],
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[scatter, surface], layout=layout)
    return fig

def plot_3d_dual_regression(rough_data, precise_data, cols):
    def fit_plane(points):
        x1, x2, y = points[:, 0], points[:, 1], points[:, 2]
        X = np.column_stack((x1, x2))
        reg = LinearRegression().fit(X, y)
        return reg, x1, x2, y

    def make_surface(reg, x1, x2, color):
        x1_range = np.linspace(x1.min(), x1.max(), 50)
        x2_range = np.linspace(x2.min(), x2.max(), 50)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)
        yy = reg.coef_[0] * xx1 + reg.coef_[1] * xx2 + reg.intercept_
        return go.Surface(
            x=xx1, y=xx2, z=yy,
            opacity=0.5,
            colorscale=color,
            showscale=False,
            name=f"Fit Plane ({color})"
        )

    reg1, x1_1, x2_1, y1 = fit_plane(rough_data)
    reg2, x1_2, x2_2, y2 = fit_plane(precise_data)

    scatter1 = go.Scatter3d(
        x=x1_1, y=x2_1, z=y1,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Rough Data'
    )
    scatter2 = go.Scatter3d(
        x=x1_2, y=x2_2, z=y2,
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Precise Data'
    )

    surface1 = make_surface(reg1, x1_1, x2_1, 'Reds')
    surface2 = make_surface(reg2, x1_2, x2_2, 'Blues')

    annotation1 = f"<b>Rough Data:</b> y = {reg1.coef_[0]:.3f}x₁ + {reg1.coef_[1]:.3f}x₂ + {reg1.intercept_:.3f}"
    annotation2 = f"<b>Precise Data:</b> y = {reg2.coef_[0]:.3f}x₁ + {reg2.coef_[1]:.3f}x₂ + {reg2.intercept_:.3f}"

    annotations = [
        dict(
            showarrow=False,
            text=annotation1,
            xref="paper", yref="paper",
            x=0.99, y=0.99,
            xanchor="right", yanchor="top",
            font=dict(size=14),
            bgcolor="rgba(255,0,0,0.1)",
            bordercolor="red",
            borderwidth=1
        ),
        dict(
            showarrow=False,
            text=annotation2,
            xref="paper", yref="paper",
            x=0.99, y=0.89,
            xanchor="right", yanchor="top",
            font=dict(size=14),
            bgcolor="rgba(0,0,255,0.1)",
            bordercolor="blue",
            borderwidth=1
        )
    ]

    layout = go.Layout(
        title="Rough vs Precise Data Regression Comparison",
        scene=dict(
            xaxis_title=cols[0] + " temp (x₁)",
            yaxis_title=cols[1] + " temp (x₂)",
            zaxis_title= "Worcester temp (y)",
        ),
        annotations=annotations,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[scatter1, scatter2, surface1, surface2], layout=layout)
    return fig

# Generate all 6 plots and save as HTML
print("Generating 6 regression plots...")

print("\n1. Rough Cambridge and Springfield Data")
fig1 = plot_3d_regression(cambridge_springfield_points_rough,
                   title="Rough Cambridge and Springfield Data",
                   cols = ("Cambridge", "Springfield"))
fig1.write_html("plot1_rough_cambridge_springfield.html")
print("Saved as plot1_rough_cambridge_springfield.html")

print("\n2. Precise Cambridge and Springfield Data")
fig2 = plot_3d_regression(cambridge_springfield_points_precise,
                   title="Precise Cambridge and Springfield Data",
                   cols = ("Cambridge", "Springfield"))
fig2.write_html("plot2_precise_cambridge_springfield.html")
print("Saved as plot2_precise_cambridge_springfield.html")

print("\n3. Comparison of Cambridge and Springfield")
fig3 = plot_3d_dual_regression(cambridge_springfield_points_rough,
                        cambridge_springfield_points_precise,
                        cols = ("Cambridge", "Springfield"))
fig3.write_html("plot3_cambridge_springfield_comparison.html")
print("Saved as plot3_cambridge_springfield_comparison.html")

print("\n4. Rough Cambridge and Boston Data")
fig4 = plot_3d_regression(cambridge_boston_points_rough, 
                   title="Rough Cambridge and Boston Data", 
                   cols = ("Cambridge", "Boston"))
fig4.write_html("plot4_rough_cambridge_boston.html")
print("Saved as plot4_rough_cambridge_boston.html")

print("\n5. Precise Cambridge and Boston Data")
fig5 = plot_3d_regression(cambridge_boston_points_precise,
                   title="Precise Cambridge and Boston Data",
                   cols = ("Cambridge", "Boston"))
fig5.write_html("plot5_precise_cambridge_boston.html")
print("Saved as plot5_precise_cambridge_boston.html")

print("\n6. Comparison of Cambridge and Boston")
fig6 = plot_3d_dual_regression(cambridge_boston_points_rough, 
                        cambridge_boston_points_precise, 
                        cols = ("Cambridge", "Boston"))
fig6.write_html("plot6_cambridge_boston_comparison.html")
print("Saved as plot6_cambridge_boston_comparison.html")

print("\nAll 6 plots generated and saved as HTML files!")
