```bash
# venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# conda
conda env create -f environment.yml
conda activate my-env

# pip-tools: compile then install
pip-compile requirements.in        # writes requirements.txt with hashes
pip-sync requirements.txt          # installs exactly that, nothing more

# poetry: install deps (creates venv automatically)
poetry install                     # reads pyproject.toml, writes/respects poetry.lock
poetry add scikit-learn            # adds dep, updates lock
poetry run python -m src.train     # run inside managed venv

# uv: drop-in pip-tools replacement
uv venv                            # create .venv (auto-detected by later uv commands)
uv pip compile requirements.in -o requirements.txt   # hash-locked, like pip-compile but faster
uv pip sync requirements.txt       # install exactly that set

# uv: full project manager (reads/writes pyproject.toml + uv.lock)
uv init my-ml-project              # scaffold a project
uv python install 3.11             # download & pin a Python version (no pyenv/conda needed)
uv add scikit-learn                # add dep, update uv.lock
uv sync                            # create venv + install exactly the locked deps
uv run python -m src.train         # run inside the managed venv

# Docker
docker build -t my-ml-project .
docker run --rm my-ml-project python -m src.pipeline.train
```