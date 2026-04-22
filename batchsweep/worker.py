import logging
import traceback
from enum import Enum
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("batchsweep")


class MsgKind(Enum):
    LOG = "log"
    PROGRESS = "progress"
    NO_MORE_JOBS = "no_more_jobs"
    RESULT = "result"
    FAILED = "failed"


class JobContext:
    """Context passed as first argument to each job function."""

    def __init__(
        self,
        device: str,
        job_id: int,
        output_dir: str,
        per_job_dir: bool,
        label: str,
        msg_queue: Queue,
    ):
        self.device = device
        self.job_id = job_id
        self._output_dir = Path(output_dir)
        self._per_job_dir = per_job_dir
        self._job_dir: Optional[Path] = None
        self._label = label
        self._msg_queue = msg_queue

    @property
    def job_dir(self) -> Path:
        if not self._per_job_dir:
            raise RuntimeError("ctx.job_dir requires per_job_dir=True")
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
    per_job_dir: bool,
) -> None:
    if device.startswith("cuda"):
        gpu_idx = device.split(":")[1]
        label = f"[GPU{gpu_idx}/W{worker_idx}]"
    else:
        label = f"[CPU/W{worker_idx}]"

    while True:
        item: Any = job_queue.get()
        if item is None:
            msg_queue.put((MsgKind.NO_MORE_JOBS, label))
            break

        job_id, job_params, job_num, total = item
        ctx = JobContext(device, job_id, output_dir, per_job_dir, label, msg_queue)

        msg_queue.put((MsgKind.PROGRESS, label, "start", job_num, total))
        try:
            result = fn(ctx, **job_params)
            msg_queue.put((MsgKind.RESULT, job_id, job_params, result))
        except Exception:
            msg_queue.put((MsgKind.FAILED, job_id, job_params, traceback.format_exc()))
        msg_queue.put((MsgKind.PROGRESS, label, "done", job_num, total))
