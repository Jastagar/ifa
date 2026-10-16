import queue
import threading
from collections.abc import Callable


class SpeechQueue:
    def __init__(
        self,
        handler: Callable[[str], None],
        maxsize: int = 32,
    ):
        self._handler = handler

        self._queue = queue.Queue(
            maxsize=maxsize
        )

        self._stop_event = threading.Event()

        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self._worker.start()


    def enqueue(self, text: str):
        if not text:
            return

        self._queue.put(text)

    def clear(self):
        while True:
            try:
                self._queue.get_nowait()

            except queue.Empty:
                break

    def stop(self):
        self._stop_event.set()

        self._queue.put(None)

    def wait_until_empty(self):
        self._queue.join()

    @property
    def size(self) -> int:
        return self._queue.qsize()


    def _run(self):
        while not self._stop_event.is_set():
            item = self._queue.get()

            if item is None:
                break

            try:
                self._handler(item)

            except Exception as e:
                print(
                    f"[SpeechQueue] worker error: {e}"
                )

            finally:
                self._queue.task_done()