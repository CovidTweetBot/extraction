import os

import pandas as pd
import plotly.graph_objects as go

from processors import RESULTS_OUTPUT_PATH


IMAGES_OUTPUT_PATH = "./output/images"


def scatter_plot(csv_file):
    df = pd.read_csv(csv_file)
    fig = go.Figure()
    c = 0
    for col in df.columns:
        if c == 0:
            first_col = col
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[first_col],
                    y=df[col],
                    name=col
                )
            )
        c += 1

    fig.update_xaxes(title=first_col)
    fig.update_yaxes(title="Valor Y")
    fig.update_layout(title=f"Datos en {csv_file}")
    # fig.show()
    return fig


def generate_visualizations():
    if not os.path.exists(IMAGES_OUTPUT_PATH):
        os.mkdir(IMAGES_OUTPUT_PATH)
    arr = os.listdir(RESULTS_OUTPUT_PATH)
    for file in arr:
        if file.endswith(".csv"):
            fig = scatter_plot(os.path.join(RESULTS_OUTPUT_PATH, file))
            fig_name = file.replace(".csv", "")
            fig.write_image(os.path.join(IMAGES_OUTPUT_PATH, f"{fig_name}.png"))
