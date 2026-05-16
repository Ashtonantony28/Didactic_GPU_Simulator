import datetime
import os


class Logger:
    def __init__(self, level: str = "info"):
        os.makedirs("test/logs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.filename = f"test/logs/log_{ts}.txt"
        self.level = level

    def _write(self, *messages):
        line = " ".join(str(m) for m in messages)
        with open(self.filename, "a") as f:
            f.write(line + "\n")

    def debug(self, *messages):
        if self.level == "debug":
            self._write(*messages)

    def info(self, *messages):
        self._write(*messages)


logger = Logger(level="info")
