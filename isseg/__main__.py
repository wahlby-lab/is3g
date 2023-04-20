from isseg.isseg import isseg
import click
import pandas as pd


@click.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("csv_out", type=click.Path(exists=False, dir_okay=False))
@click.option("--x", type=str, default="x", help="TODO")
@click.option("--y", type=str, default="y", help="TODO")
@click.option("--labels", type=str, default="labels", help="TODO")
@click.option("--radius", type=click.FloatRange(min=0), default=1, help="TODO")
@click.option("--remove_background", type=bool, default=True, help="TODO")

def main(
    csv_path: str,
    csv_out: str,
    x: str,
    y: str,
    labels: str,
    radius: float,
    remove_background: bool = True,
):
    # Load the data
    data = pd.read_csv(csv_path)

    # Run the clustering
    clusters = isseg(
        data, x=x, y=y, labels=labels, radius=radius, remove_background=remove_background
    )

    data['isseg'] = clusters
    data.to_csv(csv_out, index=False)

if __name__ == "__main__":
    main()
