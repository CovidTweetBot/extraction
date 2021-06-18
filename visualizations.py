import pathlib

import pandas as pd
import plotly.graph_objects as go


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
    output_path = pathlib.Path("./output")
    results_path = output_path / "results"
    images_path = output_path / "images"
    if not images_path.exists():
        images_path.mkdir()
    for file in results_path.glob("*.csv"):
        fig = scatter_plot(file)
        write_to = images_path / f"{file.stem}.png"
        fig.write_image(str(write_to))
