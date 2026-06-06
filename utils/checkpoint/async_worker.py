# utils/checkpoint/async_worker.py
"""Async checkpoint worker.

Manages background thread for asynchronous checkpoint writes.
"""

from __future__ import annotations

import logging
import threading
from queue import Empty, Queue
from typing import Callable

logger = logging.getLogger(__name__)


class AsyncCheckpointWorker:
    """Background thread for async checkpoint operations."""
    
    def __init__(self, max_queue_size: int = 2):
        self._queue: Queue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._max_queue_size = max_queue_size
        self._start()
    
    def _start(self) -> None:
        """Start background thread."""
        self._queue = Queue(maxsize=self._max_queue_size)
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="CheckpointAsyncWorker",
        )
        self._thread.start()
    
    def stop(self) -> None:
        """Stop background thread and wait for pending operations."""
        if self._shutdown is None:
            return
        
        self._shutdown.set()
        if self._queue:
            try:
                self._queue.put(None, timeout=1.0)
            except:
                pass
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Async worker thread did not join cleanly")
        
        self._shutdown = None
        self._queue = None
        self._thread = None
    
    def _worker_loop(self) -> None:
        """Background thread that processes async checkpoint requests."""
        while True:
            # Check shutdown status - handle case where _shutdown is set to None
            if self._shutdown is None or self._shutdown.is_set():
                break
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                
                operation, args, kwargs, callback = item
                
                try:
                    # The actual operation is performed by the caller
                    # This worker just manages the threading
                    if callback:
                        callback(None)
                except Exception as e:
                    logger.error(f"Async checkpoint operation failed: {e}")
                    if callback:
                        callback(e)
            
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Async worker error: {e}")
    
    def submit(
        self,
        operation: str,
        args: tuple = (),
        kwargs: dict | None = None,
        callback: Callable | None = None,
    ) -> None:
        """Submit an operation to the async worker.
        
        Args:
            operation: Operation name
            args: Positional arguments
            kwargs: Keyword arguments
            callback: Optional callback function
        """
        if self._queue is None:
            raise RuntimeError("Async worker not started")
        
        self._queue.put((operation, args, kwargs or {}, callback))
    
    @property
    def is_running(self) -> bool:
        """Check if the worker thread is running."""
        return self._thread is not None and self._thread.is_alive()


__all__ = ["AsyncCheckpointWorker"]
