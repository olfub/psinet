import copy
import json
import sys
from pathlib import Path

from log_redirect import PrintLogger


class PaWork:
    DEFAULT_CONFIG = {
        "name": "default",
        "description": "pawork default configuration",
        "space": {"dir": "~/.pawork/spaces/"},
    }

    # PaWork(local_space=True, env_dir="./", default_config=True)

    def __init__(
        self, exec_name, env_dir="~/.pawork/", use_default_config=True, subspace=None
    ):
        self.env_dir = Path(env_dir)

        if not self.env_dir.exists():
            raise RuntimeError("No pa work directory")
            # self.log(f"Creating pawork directory: {env_dir}")
            # self.env_dir.mkdir()
            # (self.env_dir / "spaces").mkdir()

        if not use_default_config:
            try:
                self.config = json.loads(self.env_dir / "machine.conf").read_text()
                # FIXME check config format
                self.log(f"Loading '{self.config['name']}' config.")
            except:
                use_default_config = True

        self._space_dir = Path(self.config["space"]["dir"]) / self.config["name"]

        if use_default_config:
            self.log("Could not load environment. Using default config.")
            self.config = copy.deepcopy(PaWork.DEFAULT_CONFIG)

        self._setup_print_redirect(exec_name)

    def _setup_print_redirect(self, exec_name):
        self._orig_stdout = sys.stdout
        self._log_dir = self.get_space_dir() / "logs"
        self._log_dir.mkdir(exist_ok=True)
        sys.stdout = PrintLogger(self._log_dir / f"{exec_name}.txt")

    def log(self, msg):
        print(f"[PAWORK] {msg}")

    def get_space_dir(self):
        return self._space_dir
