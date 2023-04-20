from isseg.isseg import isseg
import click
import pandas as pd


@click.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--x", type=str, default="x", help="TODO")
@click.option("--y", type=str, default="y", help="TODO")
@click.option("--labels", type=str, default="labels", help="TODO")
@click.option("--cell_diameter", type=click.FloatRange(min=0), default=1, help="TODO")
@click.option("--remove_background", type=bool, default=True, help="TODO")
def main(
    csv_path: str,
    x: str,
    y: str,
    labels: str,
    cell_diameter: float,
    remove_background: bool = True,
):
    data = pd.read_csv(csv_path)

    # Parse into numpy land
    xy = data[[x, y]].to_numpy()
    labels = data[labels].to_numpy().astype("int")

    # Run the clustering
    clusters, signed_edges, active_set = isseg(
        xy, labels, cell_diameter=cell_diameter, remove_background=remove_background
    )


if __name__ == "__main__":
    main()
