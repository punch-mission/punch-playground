# Getting Python

To get a simple local environment set up to run a notebook:

- Install [VSCode](https://code.visualstudio.com/download)

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

- Download this repository, either by manually downloading, or git cloning it

- Open a terminal and navigate into that project folder, and then execute the commands:

```sh
uv python
uv venv

source .venv/bin/activate

uv pip install punchbowl
uv pip install ipykernel
```

- Open that project folder with VSCode, open a notebook, and select that virtual environment to use.
