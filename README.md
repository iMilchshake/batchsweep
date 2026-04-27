# batchsweep

This project is motivated by me constantly getting annoyed when writing experiment scripts for ML research projects.
When my experiments are still of explorative nature I typically don't engineer the scripts with the intent to stay, so they often end up without clean multi-processing, resuming, logging or error handling.
While this allows me to produce results very fast, I often end up in frustrating situations where I have a job running for hours and wish I had implemented some of those quality of life features ahead of time, but can't as I don't want to lose the progress.
My scripts typically follow a very similar pattern, so I built this library to handle the repeating parts in an efficient and re-usable way.
There are a handful of powerful ML pipeline frameworks, but for my use cases they come with too much overhead and tend to force you into their opinionated workflows.
Therefore the philosophy of this project is to provide a clean interface, focusing on efficient orchestration of jobs while being as unrestrictive as possible.
There will not be a PyPI package _for now_. Check out my implementation and if u like it install using:

```bash
pip install git+https://github.com/iMilchshake/batchsweep.git
```

> **Alpha:** This project is in active development. I actively improve it as I use it in projects to learn about possible limitations, so use at your own risk. See [Known Limitations](#known-limitations).

## Features

- Minimalistic and unrestrictive interface
- Dynamic job scheduling with multi-processing across multiple GPUs
- Resume capabilities (skip jobs that are done already)
- Simple error handling and logging
- Optional per-job output folders

## Usage Example

```python
from batchsweep import experiment

def train(ctx, lr, dropout):
    train_data, test_data = load_data()
    model = train_model(train_data, lr=lr, dropout=dropout, device=ctx.device)
    metrics = evaluate(model, test_data)
    ctx.log(f"accuracy={metrics['accuracy']:.4f}")
    return metrics

experiment(
    fn=train,
    sweep={"lr": [1e-3, 1e-4], "dropout": [0.1, 0.3]},
    output_dir="./runs/my_sweep",
    gpus=None,  # default: use all available GPUs
    workers_per_gpu=2, # 2 processes per GPU
)
```

## Known Limitations

I might fix these in the future:

- Info about cuda devices via `torch`, assumes that projects run with torch. I might replace this with a more lightweight approach.
- No automatic dataset sharing across workers, each job needs to handle everything from dataloading to metric calculation, `batchsweep` only performs the orchestration. If dataloading requires costly pre-processing the best way most likely is to temporarily cache it. I might add some automatic handling for this (via joblib?).
- Workers are persistent so any global state changes carry over to following jobs on the same worker, users should make sure to initialize a clean local state. However, i might add a parameter that launches completely new processes instead of re-using old ones.

## API Reference

### `experiment()`

```python
def experiment(
    fn: Callable[[JobContext, ...], dict[str, float | int | str]],   # job function, return metrics dict
    sweep: Optional[dict[str, list]] = None,                         # cartesian grid (mutually exclusive with `jobs`)
    jobs: Optional[list[dict[str, Any]]] = None,                     # explicit job list (mutually exclusive with `sweep`)
    output_dir: str | Path,                                          # run directory, created if missing
    gpus: Optional[list[int]] = None,                                # GPU indices, None = use all available
    workers_per_gpu: int = 1,                                        # concurrent workers per GPU
    on_job_done: Optional[Callable[[int, dict, dict], None]] = None, # called after each finished job, (job_id, params, metrics) -> None
    on_job_fail: Optional[Callable[[int, dict, str], None]] = None,  # called after each failed job, (job_id, params, traceback_str) -> None
    on_complete: Optional[Callable[[], None]] = None,                # called after all jobs finish, () -> None
) -> None
```

### `JobContext`

```python
class JobContext:
    device: str               # e.g. "cuda:0" or "cpu"
    job_id: int               # unique ID
    job_dir: Path             # optional per-job output directory, is created lazily
    def log(msg: str) -> None # send log message to the main process logger
```

### Output structure

```
output_dir/
├── results.csv    # job_id, status, <params...>, <metrics...>
├── experiment.log # timestamped log of all activity
├── failures.log   # tracebacks of failed jobs
└── jobs/          # created on first ctx.job_dir access
    ├── 0/
    ├── 1/
    └── ...
```

## Run Example

1. Clone repo and install locally:

```bash
git clone https://github.com/iMilchshake/batchsweep.git
cd batchsweep
pip install -e ".[dev]"
```

2. Generate example dataset, then run the test sweep script:

```bash
cd examples
python generate_dataset.py
python xgb_sweep.py
```
