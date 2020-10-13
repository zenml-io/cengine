# cengine
This is the maiot Core Engine. Published pip package can be found [here](https://pypi.org/project/cengine).

The Core Engine is a platform that lets you create machine learning pipelines for production use-cases.
The [maiot website](https://maiot.io) gives an overview of the features of the Core Engine.
For more information on how to use the Core Engine, please refer to the [docs](https://docs.maiot.io).

## How to install from pip
```bash
pip install cengine
```

## How to install from source
```bash
make venv
source venv/bin/activate
make install
make build
```

## Known errors in installation
If you run into a `psutil` error, please install the python-dev libraries:

```bash
sudo apt update
sudo apt install python3.x-dev
```

## Enabling auto completion on the CLI

For Bash, add this to ~/.bashrc:
```bash
eval "$(_CENGINE_COMPLETE=source_bash cengine)"
```

For Zsh, add this to ~/.zshrc:
```bash
eval "$(_CENGINE_COMPLETE=source_zsh cengine)"
```

For Fish, add this to ~/.config/fish/completions/foo-bar.fish:
```bash
eval (env _CENGINE_COMPLETE=source_fish cengine)
```

## Authors

* **maiot GmbH** - [maiot.io](https://maiot.io) - [maiot Docs](https://docs.maiot.io)