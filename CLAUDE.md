# TUPANN

Satellite-only precipitation nowcasting (GOES-16 RRQPE, IMERG) with PyTorch Lightning. `README.md` has the commands for downloading data, processing it, generating splits, training and evaluating.

Run everything from the repo root with `PYTHONPATH=$PWD`, because scripts import `src.*` and open `configs/...`, `data/...` and `models/...` by relative paths.

## Configs drive everything

`src/models/train.py` and `src/models/predict.py` take three config **names** (file stems, without paths):

- `--data_config` → `configs/data/<name>.yaml`: the input/target datasets with their lags (in minutes) and transforms, plus a block per `train`/`val`/`test` split holding `locations` and a `datetimes` file.
- `--eval_config` → `configs/eval/<name>.yaml`: the monitored metric (`CSI8` is maximized; anything else is minimized) and how many checkpoints to keep.
- `--hparams_config` → `configs/models/<name>.yaml`: `model_name` plus the model kwargs. A list-valued hparam becomes a grid search: `train.py` trains every combination in its cartesian product.

`model_name` selects `src/models/<model_name>/lightning.py:model`. When that module doesn't exist, `<main>_<sub>` falls back to `src/models/<main>/<sub>_lightning.py`. For example, `tupann_autoenc` resolves to `src/models/tupann/autoenc_lightning.py`. Every model subclasses `LModule` in `src/models/lightning.py`.

## Model hash: hparams are the identity

`get_model_path` (`src/utils/model_utils.py`) names the output directory `models/<model_name>/<md5 of every hparam>/`. The consequences:

- `predict.py` and the dataset keys described below find a trained model by **recomputing** that hash from the config. Once a run has trained, freeze its `configs/models/*.yaml`: changing, adding or removing any key (even `n_epochs`) points at a new, empty directory. Put a new variant in a new config file.
- A new constructor kwarg needs a default that preserves the old behavior, and must stay out of existing configs.
- Checkpoints live at `models/<model_path>/train/<data_config>/<code>--best-epoch=N.ckpt`. `h_params.yaml` and `ckpt_tree.jsonl` are written there only **after** `trainer.fit` returns.
- Two-stage models (TUPANN, rescast) read the stage-1 `h_params.yaml` through `autoenc_hash`/`autoenc_ckpt`. If you stop a stage-1 run before it finishes (in Lightning 2.0, SIGTERM crashes it before the save), write `h_params.yaml` (the config's hparams) and `ckpt_tree.jsonl` by hand before starting stage 2.

## Dataset keys

A key in a data config's `input`/`target` is either a raw dataset (`goes16_rrqpe`, `imerg`) or a `#`-separated reference, resolved by `DatasetHandlerFactory` in `src/data/dataset_handlers.py`:

- `fields_intensities#goes16_rrqpe#darts` (or `#lk`): precomputed optical-flow motion fields and intensities, used as TUPANN's supervision targets.
- `<model>#<data_config>#<hash>[#<ckpt stem>]`: the saved predictions of another trained model, used as an input. NowcastNet uses this to consume EvoNet outputs. `<model>` must be registered in `DatasetHandlerFactory._handlers`, and its predictions must already exist (run `predict.py` on that model for every split it needs).

Inside the batch dict, a key is shortened to its first `#` segment.

Standardization statistics come from `data/transform_params/<dataset>.<train datetimes file stem>.yaml`, computed on the **train** split. Changing the train datetimes file means those statistics have to exist for the new file.

## Datetime files

`configs/data/{split}_datetimes[a:-b]-goes16_rrqpe-rj.txt` lists the valid start times of each split. The first `a` and the last `b` steps of every rain event are dropped so the input lags and target lead times stay inside the event. Pick `b` to match the longest target lead time: `[10:-18]` covers 180 min at 10-min steps, and `[10:-1]` covers single-step targets. `_small`/`_sample` files are subsets for smoke tests.

## Repo layout outside `src/`

These folders are gitignored and machine-local: `data*/` (raw, processed, transform params, sample weights), `models/` (checkpoints, logs, predictions), `eval/` (metric CSVs), `personal/` (scratch notebooks, launch scripts, logs) and `model_names.jsonl` (an append-only log mapping hashes to hparams, written by every run). Commit only `src/`, `configs/`, `README.md` and the dependency files.

## Style

Black and isort with a line length of 120 (`pyproject.toml`). The repo has no test suite; to check a change, run a short training with a `_sample` datetimes file, or `predict.py` on an existing checkpoint.
