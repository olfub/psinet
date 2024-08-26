import atexit
import sys


class PrintLogger:
    def __init__(self, log_path, verbose=True):
        self._log_path = log_path
        self.terminal = sys.stdout
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        print(f"Logging to {log_path}")

        # open log file
        self._log_path.parent.mkdir(exist_ok=True, parents=True)
        self._log_file = open(self._log_path, "w+")
        sys.stdout = self

        # setup hook to close file
        # sys.excepthook(type, value, traceback)
        atexit.register(self._close_log_file)

    def write(self, message):
        self.terminal.write(message)
        self._log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self._log_file.flush()

    def close(self):
        self._close_log_file()

    def _close_log_file(self):
        try:
            pass
            self.flush()
            self._log_file.close()
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except:
            pass
