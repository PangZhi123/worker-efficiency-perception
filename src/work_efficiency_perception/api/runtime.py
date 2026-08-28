from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, RLock
from typing import Protocol

from .repository import (
    TERMINAL_TASK_STATUSES,
    ConfigurationRecord,
    InMemoryRepository,
    TaskRecord,
    TaskStatus,
)


class AnalysisWorker(Protocol):
    def run(
        self,
        task: TaskRecord,
        configuration: ConfigurationRecord,
        stop_event: Event,
    ) -> None: ...


@dataclass(slots=True)
class _Execution:
    stop_event: Event
    future: Future[None]


class TaskManager:
    def __init__(
        self,
        repository: InMemoryRepository,
        worker: AnalysisWorker,
        *,
        timestamp: Callable[[], str],
        max_workers: int = 2,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        self._repository = repository
        self._worker = worker
        self._timestamp = timestamp
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="work-efficiency"
        )
        self._lock = RLock()
        self._executions: dict[str, _Execution] = {}

    def start(self, task_id: str) -> TaskRecord:
        task = self._repository.transition_task(
            task_id,
            expected={TaskStatus.CREATED},
            status=TaskStatus.STARTING,
            updated_at=self._timestamp(),
        )
        if task is None:
            current = self._repository.get_task(task_id)
            if current is None:
                raise KeyError(task_id)
            return current
        configuration = self._repository.get_configuration(task.configuration_id)
        if configuration is None:
            failed = self._repository.transition_task(
                task_id,
                expected={TaskStatus.STARTING},
                status=TaskStatus.FAILED,
                updated_at=self._timestamp(),
                failure_code="WE0202",
                failure_message="configuration disappeared before task start",
            )
            return failed or task

        stop_event = Event()
        with self._lock:
            future = self._executor.submit(self._execute, task, configuration, stop_event)
            self._executions[task_id] = _Execution(stop_event, future)
        return self._repository.get_task(task_id) or task

    def request_stop(self, task_id: str) -> TaskRecord:
        task = self._repository.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.status in TERMINAL_TASK_STATUSES:
            return task
        with self._lock:
            execution = self._executions.get(task_id)
            if execution is not None:
                execution.stop_event.set()
                if execution.future.cancel():
                    stopped = self._repository.transition_task(
                        task_id,
                        expected={TaskStatus.STARTING},
                        status=TaskStatus.STOPPED,
                        updated_at=self._timestamp(),
                    )
                    self._executions.pop(task_id, None)
                    return stopped or self._repository.get_task(task_id) or task
        stopping = self._repository.transition_task(
            task_id,
            expected={TaskStatus.CREATED, TaskStatus.STARTING, TaskStatus.RUNNING},
            status=TaskStatus.STOPPING,
            updated_at=self._timestamp(),
        )
        return stopping or self._repository.get_task(task_id) or task

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            for task_id, execution in list(self._executions.items()):
                execution.stop_event.set()
                if execution.future.cancel():
                    self._repository.transition_task(
                        task_id,
                        expected={TaskStatus.STARTING},
                        status=TaskStatus.STOPPED,
                        updated_at=self._timestamp(),
                    )
                    self._executions.pop(task_id, None)
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(
        self,
        task: TaskRecord,
        configuration: ConfigurationRecord,
        stop_event: Event,
    ) -> None:
        try:
            running = self._repository.transition_task(
                task.task_id,
                expected={TaskStatus.STARTING},
                status=TaskStatus.RUNNING,
                updated_at=self._timestamp(),
            )
            if running is None:
                if stop_event.is_set():
                    self._repository.transition_task(
                        task.task_id,
                        expected={TaskStatus.STOPPING},
                        status=TaskStatus.STOPPED,
                        updated_at=self._timestamp(),
                    )
                return
            self._worker.run(running, configuration, stop_event)
            final_status = TaskStatus.STOPPED if stop_event.is_set() else TaskStatus.COMPLETED
            self._repository.transition_task(
                task.task_id,
                expected={TaskStatus.RUNNING, TaskStatus.STOPPING},
                status=final_status,
                updated_at=self._timestamp(),
            )
        except Exception as error:  # noqa: BLE001 - worker failures become task failures
            if stop_event.is_set():
                self._repository.transition_task(
                    task.task_id,
                    expected={TaskStatus.STARTING, TaskStatus.RUNNING, TaskStatus.STOPPING},
                    status=TaskStatus.STOPPED,
                    updated_at=self._timestamp(),
                )
                return
            self._repository.transition_task(
                task.task_id,
                expected={TaskStatus.STARTING, TaskStatus.RUNNING, TaskStatus.STOPPING},
                status=TaskStatus.FAILED,
                updated_at=self._timestamp(),
                failure_code="WE0301",
                failure_message=str(error),
            )
        finally:
            with self._lock:
                self._executions.pop(task.task_id, None)
