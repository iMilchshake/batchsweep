# batchsweep

Motivation: this project is motivated by me constantly getting annoyed when writing experiment scripts for ML research projects.
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
    params={"lr": [1e-3, 1e-4], "dropout": [0.1, 0.3]},
    output_dir="./runs/my_sweep",
    gpus=None,  # default: use all available GPUs
    workers_per_gpu=2, # 2 processes per GPU
)
```

## Features

- Minimalistic and unrestrictive interface
- Dynamic job scheduling with multi-processing across multiple GPUs
- Resume capabilities (skip jobs that are done already)
- Simple error handling and logging
- Optional per-job output folders

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

## Known Limitations

I might fix these in the future:

- Info about cuda devices via `torch`, assumes that projects run with torch
- No automatic dataset sharing across workers, each job needs to handle everything from dataloading to metric calculation, `batchsweep` only performs the orchestraction
