# IS3G

In Situ Sequencing Segmentation

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install is3g:

    pip install is3g

## Usage

See [example.ipynb](this) Jupyter Notebook for a minimal example.

You can also use the `is3g` command to execute is3g from the command-line. For example:
```console
$ is3g examples/data.csv examples/data_out.csv --x X --y Y --label Gene --radius 22.5
Training on device: cpu
Epoch 222: loss=0.461070, accuracy=0.784571:
44%|████████████████████████████                                         | 221/500 [00:06<00:08, 34.86it/s]
Stopping early at epoch 221 with loss 0.4128
```

You can list all options with:

```console
$ is3g --help
Usage: is3g [OPTIONS] CSV_PATH CSV_OUT

Options:
  -x, --x TEXT                    Column in the CSV file where the x coordinate is stored for each gene.

  -y, --y TEXT                    Column in the CSV file where the y coordinate is stored for each gene.

  -l, --label TEXT                Column in the CSV file where the gene labels are stored.
  
  -r, --radius FLOAT RANGE        Approximate radius of a cell.  [x>=0]
  
  --remove-background / --no-remove-background
                                  Specify whether to automatically remove genes from low-density regions. Default is true.
  
  --version                       Show the version and exit. 
  
  --help                          Show this message and exit.
```


## Support

If (when) you find a bug, please [raise an issue](https://github.com/wahlby-lab/is3g/issues/new).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Authors

[Axel Andersson](mailto:axel.andersson@it.uu.se)

[Andrea Behanova](mailto:andrea.behanova@it.uu.se)

## License

[BSD 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)
