import click
import pandas as pd

from isseg import __version__ as version
from isseg.isseg import isseg


@click.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("csv_out", type=click.Path(exists=False, dir_okay=False))
@click.option("-x", "--x", type=str, default="x", help="Column in the CSV file where the x coordinate is stored for each gene.")
@click.option("-y", "--y", type=str, default="y", help="Column in the CSV file where the y coordinate is stored for each gene.")
@click.option("-l", "--label", type=str, default="label", help="Column in the CSV file where the label of each gene is stored.")
@click.option("-r", "--radius", type=click.FloatRange(min=0), default=1, help="Approximate radius of a cell.")
@click.option("--remove-background/--no-remove-background", default=True, help="Specify whether to remove background genes or not. Default is true.")
@click.version_option(version=version)
def main(
    csv_path: str,
    csv_out: str,
    x: str,
    y: str,
    label: str,
    radius: float,
    remove_background: bool = True,
):
    # Load the data
    data = pd.read_csv(csv_path)

    # Run the clustering
    clusters = isseg(
        data,
        x=x,
        y=y,
        label=label,
        radius=radius,
        remove_background=remove_background,
    )

    data["isseg"] = clusters
    data.to_csv(csv_out, index=False)


if __name__ == "__main__":
    main()
