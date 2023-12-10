import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def multi_axis_line_plot(df, columns, title=None, x_axis_title="Time", y_axis_title=None):
    # Create a subplot with secondary_y for each column
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col in columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col, mode='markers+lines', marker_symbol='cross'),
            secondary_y=True,
        )

    # Update layout
    if title:
        fig.update_layout(title=title)
    if y_axis_title:
        fig.update_yaxes(title_text=y_axis_title, secondary_y=False)
    if x_axis_title:
        fig.update_xaxes(title_text=x_axis_title)

    fig.show()


def xy_scatter_plot(df, x_col, y_col, title=None, x_axis_title=None, y_axis_title=None):
    # Ensure the columns are numeric
    df = df[[x_col, y_col]].dropna().astype(float)

    # Calculate the linear regression
    slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
    line = slope * df[x_col] + intercept

    # Calculate the R-squared value
    correlation_matrix = np.corrcoef(df[x_col], df[y_col])
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2

    # Create the scatter plot with Plotly
    fig = go.Figure()

    # Add scatter plot for the data points
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name='Data', marker_symbol='cross'))

    # Add regression line
    fig.add_trace(go.Scatter(x=df[x_col], y=line, mode='lines', name=f'Linear Regression (R\u00b2 = {r_squared:.2f})'))

    # Update layout
    
    fig.update_layout(
        title=f"Scatter Plot of {x_col} vs {y_col} with Regression Line",
        xaxis_title=x_col,
        yaxis_title=y_col
    )

    # Show the figure
    fig.show()