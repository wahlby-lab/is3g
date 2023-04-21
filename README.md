# isseg

In Situ Sequencing Segmentation

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install isseg:

    pip install isseg

## Usage

Jupyter notebook examples using the `isseg` python API can be found in the [examples](examples) directory.

You can also use the `isseg` command to execute isseg from the command-line. For example:
```console
$ isseg examples/data.csv examples/data_out.csv --x X --y Y --label Gene --radius 22.5
Training on device: cpu
Epoch 222: loss=0.461070, accuracy=0.784571:
44%|████████████████████████████                                         | 221/500 [00:06<00:08, 34.86it/s]
Stopping early at epoch 221 with loss 0.4128
```

You can list all options with:

```console
$ isseg --help
Usage: isseg [OPTIONS] CSV_PATH CSV_OUT

Options:
  -x, --x TEXT                    TODO
  -y, --y TEXT                    TODO
  -l, --label TEXT                TODO
  -r, --radius FLOAT RANGE        TODO  [x>=0]
  --remove-background / --no-remove-background
                                  TODO
  --version                       Show the version and exit. 
  --help                          Show this message and exit.
```

## Support

If you find a bug, please [raise an issue](https://github.com/wahlby-lab/isseg/issues/new).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Authors

[Axel Andersson](mailto:axel.andersson@it.uu.se)

[Andrea Behanova](mailto:andrea.behanova@it.uu.se)

## License

[BSD 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)
