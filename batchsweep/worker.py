import logging
import traceback
from enum import Enum
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("batchsweep")


class MsgKind(Enum):
    LOG = "log"
    PROGRESS = "progress"
    NO_MORE_JOBS = "no_more_jobs"
    RESULT = "result"
    FAILED = "failed"


class SharedData:
    def __init__(self, shared_dir: Path):
        self._dir = shared_dir
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.load(self._dir / f"{key}.npy", mmap_mode="r")
        return self._cache[key]


class JobContext:
    """Context passed as first argument to each job function."""

    def __init__(
        self,
        device: str,
        job_id: int,
        output_dir: str,
        label: str,
        msg_queue: Queue,
        shared_data: Optional["SharedData"] = None,
    ):
        self.device = device
        self.job_id = job_id
        self.shared_data = shared_data
        self._output_dir = Path(output_dir)
        self._job_dir: Optional[Path] = None
        self._label = label
        self._msg_queue = msg_queue

    @property
    def job_dir(self) -> Path:
        if self._job_dir is None:
            self._job_dir = self._output_dir / "jobs" / str(self.job_id)
            self._job_dir.mkdir(parents=True, exist_ok=True)
        return self._job_dir

    def log(self, msg: str) -> None:
        self._msg_queue.put((MsgKind.LOG, f"{self._label} {msg}"))


def _worker_fn(
    fn: Callable,
    job_queue: Queue,
    msg_queue: Queue,
    device: str,
    worker_idx: int,
    output_dir: str,
    shared_dir: Optional[str] = None,
) -> None:
    if device.startswith("cuda"):
        gpu_idx = device.split(":")[1]
        label = f"[GPU{gpu_idx}/W{worker_idx}]"
    else:
        label = f"[CPU/W{worker_idx}]"

    shared_data = SharedData(Path(shared_dir)) if shared_dir is not None else None

    while True:
        item: Any = job_queue.get()
        if item is None:
            msg_queue.put((MsgKind.NO_MORE_JOBS, label))
            break

        job_id, job_params, job_num, total = item
        ctx = JobContext(device, job_id, output_dir, label, msg_queue, shared_data)

        msg_queue.put((MsgKind.PROGRESS, label, "start", job_num, total, job_id))
        try:
            result = fn(ctx, **job_params)
            msg_queue.put((MsgKind.RESULT, job_id, job_params, result))
        except Exception:
            msg_queue.put((MsgKind.FAILED, job_id, job_params, traceback.format_exc()))
        msg_queue.put((MsgKind.PROGRESS, label, "done", job_num, total, job_id))
