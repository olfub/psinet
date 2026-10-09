import argparse
import copy
import json
import logging
import os
import threading
import time

import requests
import torch
import uvicorn
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.read_and_write import json as cs_json
from fastapi import FastAPI
from pydantic import BaseModel

from examples.synthetic.run_einet import build_parser, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [einet_hpo] %(message)s")
logger = logging.getLogger(__name__)

FAILED_PERFORMANCE = -1e10


def get_search_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformFloatHyperparameter("lr", 1e-5, 1e-2, log=True),
        UniformIntegerHyperparameter("batch_size", 16, 512, log=True),
        UniformIntegerHyperparameter("num_epochs", 5, 100),
        UniformIntegerHyperparameter("depth", 2, 12),
        UniformIntegerHyperparameter("K", 2, 20),
        CategoricalHyperparameter("nn", ["mlp", "cnn"]),
        CategoricalHyperparameter("hidden_dims", ["64", "128", "256", "128,128", "256,256"]),
    ])
    return cs


class TestConfigRequest(BaseModel):
    run_id: str
    config: dict


class TrialRunner:

    def __init__(self, base_args, hpo_dir):
        self.base_args = base_args
        self.hpo_dir = hpo_dir
        self.log_file = os.path.join(hpo_dir, "hpo_log.jsonl")
        self.trial = 0
        self.lock = threading.Lock()

    def make_args(self, config, trial):
        args = copy.deepcopy(self.base_args)
        for key, value in config.items():
            default = getattr(args, key)
            # some optimizers propose floats for integer hyperparameters
            setattr(args, key, int(round(value)) if isinstance(default, int) else type(default)(value))
        args.save_path = os.path.join(self.hpo_dir, f"trial_{trial}")
        return args

    def __call__(self, run_id, config):
        with self.lock:
            trial = self.trial
            self.trial += 1
            args = self.make_args(config, trial)
            logger.info(f"Trial {trial}: testing config {config}")
            start = time.time()
            try:
                einet = train(args)
                del einet
                with open(os.path.join(args.save_path, "results.txt")) as f:
                    results = json.load(f)
                performance = -min(results["nll_test"])
                status = "ok"
            except Exception as e:
                logger.exception(f"Trial {trial} failed")
                performance = FAILED_PERFORMANCE
                status = f"failed: {e}"
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            entry = {
                "run_id": run_id,
                "trial": trial,
                "config": config,
                "val_performance": performance,
                "status": status,
                "duration": time.time() - start,
            }
            logger.info(f"Trial {trial}: performance {performance} ({status})")
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return {"val_performance": performance, "cost": entry["duration"]}


def create_app(runner):
    app = FastAPI(title="psinet einet HPO client")

    # sync handler so training runs in a worker thread
    @app.post("/test_config")
    def test_config(req: TestConfigRequest):
        return runner(req.run_id, req.config)

    return app


def start_server(app, host, port):
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.1)
    return server, thread


def main():
    parser = argparse.ArgumentParser(
        description="Run HPO for run_einet.py via mc-hpo. Unknown arguments are forwarded to run_einet.py."
    )
    parser.add_argument("--mchpo-url", default="http://mchpo:8000")
    parser.add_argument("--callback-url", default="http://psinet:8001/test_config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--optimizer", default="rs")
    parser.add_argument("--optimizer-config", default='{"iterations": 20}', type=json.loads)
    parser.add_argument("--hpo-dir", default="results/hpo")
    hpo_args, einet_argv = parser.parse_known_args()
    base_args = build_parser().parse_args(einet_argv)

    os.makedirs(hpo_args.hpo_dir, exist_ok=True)
    runner = TrialRunner(base_args, hpo_args.hpo_dir)
    server, thread = start_server(create_app(runner), hpo_args.host, hpo_args.port)

    response = requests.post(f"{hpo_args.mchpo_url}/new_run", json={
        "optimizer_name": hpo_args.optimizer,
        "optimizer_config": hpo_args.optimizer_config,
        "search_space": cs_json.write(get_search_space()),
        "callback_url": hpo_args.callback_url,
    })
    response.raise_for_status()
    run_id = response.json()["run_id"]
    logger.info(f"Created HPO run {run_id}")

    response = requests.post(f"{hpo_args.mchpo_url}/optimize", json={"run_id": run_id}, timeout=None)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Best config: {result['best_config']} with performance {result['best_performance']}")
    with open(os.path.join(hpo_args.hpo_dir, f"result_{run_id}.json"), "w") as f:
        json.dump(result, f, indent=2)

    server.should_exit = True
    thread.join()


if __name__ == "__main__":
    main()
