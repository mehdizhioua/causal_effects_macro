import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def multi_axis_line_plot(df, columns):
    # Create a subplot with secondary_y for each column
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col in columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col, mode='markers+lines', marker_symbol='cross'),
            secondary_y=True,
        )

    # Update layout
    fig.update_layout(title="Multi-Axis Line Plot")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Values", secondary_y=True)

    fig.show()

def xy_scatter_plot(df, x_col, y_col):
    fig = go.Figure(data=go.Scatter(x=df[x_col], y=df[y_col], mode='markers', marker_symbol='cross'))
    fig.update_layout(title=f"Scatter Plot of {x_col} vs {y_col}", xaxis_title=x_col, yaxis_title=y_col)
    fig.show()