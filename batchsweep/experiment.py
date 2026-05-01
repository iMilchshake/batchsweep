import csv
import logging
import signal
import time
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch

from .worker import MsgKind, _worker_fn, logger


def _setup_logger(log_file: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _load_existing(
    results_path: Path, param_keys: list[str]
) -> tuple[set, int, Optional[list[str]]]:
    if not results_path.exists():
        return set(), 0, None

    with open(results_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return set(), 0, None

    param_set = set(param_keys)
    csv_params = {k for k in fieldnames if k in param_set}
    if csv_params != param_set:
        raise ValueError(
            f"Param keys mismatch. CSV has {sorted(csv_params)}, "
            f"current params have {sorted(param_set)}. Use a new output_dir."
        )

    metric_keys = [
        k for k in fieldnames if k not in {"job_id", "status"} and k not in param_set
    ]
    done = {tuple(str(row[k]) for k in param_keys) for row in rows}
    next_id = max(int(row["job_id"]) for row in rows) + 1
    return done, next_id, metric_keys


def _write_header(path: Path, param_keys: list[str], metric_keys: list[str]) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["job_id", "status"] + param_keys + metric_keys)


def _append_row(
    path: Path,
    job_id: int,
    status: str,
    job_params: dict,
    param_keys: list[str],
    metric_keys: list[str],
    metrics: Optional[dict] = None,
) -> None:
    with open(path, "a", newline="") as f:
        row = [job_id, status]
        row += [job_params[k] for k in param_keys]
        row += [metrics.get(k, "") if metrics else "" for k in metric_keys]
        csv.writer(f).writerow(row)


def _append_failure(path: Path, job_id: int, job_params: dict, tb: str) -> None:
    with open(path, "a") as f:
        f.write(f"\n--- Job {job_id} FAILED ---\n")
        f.write(
            "Params: " + ", ".join(f"{k}={v}" for k, v in job_params.items()) + "\n"
        )
        f.write(tb + "\n")


def _fmt_eta(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"~{s}s"
    elif s < 3600:
        return f"~{s // 60}m {s % 60}s"
    else:
        return f"~{s // 3600}h {(s % 3600) // 60}m"


def experiment(
    fn: Callable,
    output_dir: Union[str, Path],
    sweep: Optional[dict[str, list]] = None,
    jobs: Optional[list[dict[str, Any]]] = None,
    gpus: Optional[list[int]] = None,
    workers_per_gpu: int = 1,
    on_job_done: Optional[Callable[[int, dict, dict], None]] = None,
    on_job_fail: Optional[Callable[[int, dict, str], None]] = None,
    on_complete: Optional[Callable[[], None]] = None,
) -> None:
    if (sweep is None) == (jobs is None):
        raise ValueError("Exactly one of `sweep` or `jobs` must be provided.")

    if sweep is not None:
        param_keys = list(sweep.keys())
        all_job_params = [
            dict(zip(param_keys, combo)) for combo in product(*sweep.values())
        ]
    else:
        if not jobs:
            raise ValueError("`jobs` must not be empty.")
        param_keys = list(jobs[0].keys())
        expected = set(param_keys)
        for i, j in enumerate(jobs[1:], 1):
            if set(j.keys()) != expected:
                raise ValueError(
                    f"`jobs[{i}]` has keys {sorted(j.keys())}, expected {sorted(expected)}."
                )
        all_job_params = jobs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logger(output_dir / "experiment.log")
    results_path = output_dir / "results.csv"
    failures_path = output_dir / "failures.log"

    done, next_id, existing_metrics = _load_existing(results_path, param_keys)

    pending = []
    for job_params in all_job_params:
        if tuple(str(job_params[k]) for k in param_keys) not in done:
            pending.append((next_id, job_params))
            next_id += 1

    if not pending:
        logger.info("All jobs already completed.")
        return

    total = len(pending)
    skipped = len(all_job_params) - total
    logger.info(
        f"Running {total} jobs" + (f" ({skipped} skipped)." if skipped else ".")
    )

    if gpus is None:
        gpus = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )

    slots = (
        [(f"cuda:{g}", w) for g in gpus for w in range(workers_per_gpu)]
        if gpus
        else [("cpu", w) for w in range(workers_per_gpu)]
    )

    mp = get_context("spawn")
    job_q = mp.Queue()
    msg_q = mp.Queue()

    for i, (job_id, job_params) in enumerate(pending):
        job_q.put((job_id, job_params, i + 1, total))
    for _ in slots:
        job_q.put(None)

    procs = [
        mp.Process(
            target=_worker_fn,
            args=(fn, job_q, msg_q, g, w, str(output_dir)),
        )
        for g, w in slots
    ]
    for p in procs:
        p.start()

    interrupted = False
    orig_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(_sig, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupted. Waiting for current jobs to finish...")

    signal.signal(signal.SIGINT, _on_sigint)

    metric_keys = existing_metrics
    buffered_failed: list[tuple[int, dict]] = []
    done_count = 0
    start_time = time.time()

    while done_count < total:
        try:
            msg = msg_q.get(timeout=1.0)
        except Exception:
            if interrupted or all(not p.is_alive() for p in procs):
                break
            continue

        kind = msg[0]

        if kind == MsgKind.LOG:
            logger.info(msg[1])

        elif kind == MsgKind.PROGRESS:
            _, label, event, job_num, job_total, job_id = msg
            if event == "start":
                logger.info(f"{label} Start job {job_num}/{job_total} (idx={job_id})")
            else:
                elapsed = time.time() - start_time
                avg_per_job = elapsed / done_count if done_count else elapsed
                remaining = (total - done_count) * avg_per_job
                eta = _fmt_eta(remaining)
                logger.info(
                    f"{label} Finished job {job_num}/{job_total} (idx={job_id}) [ETA {eta}]"
                )

        elif kind == MsgKind.NO_MORE_JOBS:
            logger.info(f"{msg[1]} No more jobs")

        elif kind == MsgKind.RESULT:
            _, job_id, job_params, result = msg
            if metric_keys is None:
                overlap = set(result.keys()) & (set(param_keys) | {"job_id", "status"})
                if overlap:
                    raise ValueError(
                        f"Metric keys {sorted(overlap)} conflict with reserved/parameter keys. "
                        f"Rename these in your return dict."
                    )
                metric_keys = list(result.keys())
                _write_header(results_path, param_keys, metric_keys)
                for fid, fparams in buffered_failed:
                    _append_row(
                        results_path, fid, "failed", fparams, param_keys, metric_keys
                    )
                buffered_failed.clear()
            elif set(result.keys()) != set(metric_keys):
                missing = set(metric_keys) - set(result.keys())
                extra = set(result.keys()) - set(metric_keys)
                raise ValueError(
                    f"Job {job_id} metric key mismatch. "
                    f"Missing: {sorted(missing) or 'none'}, "
                    f"Extra: {sorted(extra) or 'none'}"
                )
            _append_row(
                results_path,
                job_id,
                "success",
                job_params,
                param_keys,
                metric_keys,
                result,
            )
            done_count += 1
            if on_job_done is not None:
                on_job_done(job_id, job_params, result)

        elif kind == MsgKind.FAILED:
            _, job_id, job_params, tb = msg
            _append_failure(failures_path, job_id, job_params, tb)
            if metric_keys is not None:
                _append_row(
                    results_path, job_id, "failed", job_params, param_keys, metric_keys
                )
            else:
                buffered_failed.append((job_id, job_params))
            done_count += 1
            if on_job_fail is not None:
                on_job_fail(job_id, job_params, tb)

    signal.signal(signal.SIGINT, orig_sigint)

    if interrupted:
        time.sleep(2)  # grace period so workers can finish writing before SIGTERM
        for p in procs:
            p.terminate()
    elif on_complete is not None:
        on_complete()

    for p in procs:
        p.join()
