# HeuristicBoundaryEncoding
An implementation of the heuristic boundary neural quantum state.

## Setup
This project uses `uv` for environment and dependency management.

1. Install `uv` if you don’t have it yet.
2. Create a virtual environment: `uv venv`
3. Sync dependencies (pick exactly one extra):
4. `uv sync --extra cpu --group dev`
5. `uv sync --extra cu128 --group dev`
6. `uv sync --extra cu129 --group dev`
7. `uv sync --extra cu130 --group dev`
8. (Optional) Install pre-commit hooks: `uv run pre-commit install`
9. Verify torch: `uv run python -c "import torch; print(torch.__version__)"`

Use only one of the extras at a time.

### Extras
- `cpu`: CPU-only PyTorch wheels.
- `cu128`: PyTorch wheels built for CUDA 12.8.
- `cu129`: PyTorch wheels built for CUDA 12.9.
- `cu130`: PyTorch wheels built for CUDA 13.0.

### Dependency Groups
- `dev`: Developer tools (tox, pytest, ruff, mypy, pre-commit).

## Development
### Test
- Run all checks via tox: `uv run tox`
- Run tests only: `uv run pytest`

### Lint & Format
- Lint with ruff: `uv run ruff check src`
- Auto-fix lint issues: `uv run ruff check src --fix`
- Format code: `uv run ruff format src`
