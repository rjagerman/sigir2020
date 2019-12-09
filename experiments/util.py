import json
import logging
import os
from tempfile import NamedTemporaryFile
from time import time

import torch
from pytorchltr.dataset.svmrank import svmranking_dataset as _load
from joblib.memory import Memory as _Memory
from ignite.engine import Events

load_dataset = _Memory(".cache", compress=6).cache(_load)


LOGGER = logging.getLogger(__name__)


def get_torch_device(enable_cuda):
    """Gets the torch device to use.

    Arguments:
        enable_cuda: Boolean to indicate whether to use CUDA.

    Returns:
        The torch device.
    """
    if enable_cuda and torch.cuda.is_available():
        out = torch.device("cuda")
    else:
        out = torch.device("cpu")
    LOGGER.info("Using device %s", out)
    return out


def every_n_iteration(engine, n, callback):
    def _callback_fn(_engine):
        if engine.state.iteration % n == 0:
            return callback(_engine)
    return _callback_fn


class JsonLogger:
    def __init__(self, output_file, indent=None):
        self._output_file = output_file
        self._output = {}
        self._indent = indent

    def append(self, key, value):
        next_dict = self._output
        keys = key.split("/")
        for k in keys[:-1]:
            if k not in next_dict:
                next_dict[k] = {}
            next_dict = next_dict[k]
        if keys[-1] not in next_dict:
            next_dict[keys[-1]] = []
        next_dict[keys[-1]].append(value)

    def write_to_disk(self):
        # Performs an atomic write to file.
        dirpath, filename = os.path.split(self._output_file)
        with NamedTemporaryFile(dir=dirpath, prefix=filename, mode='wt',
                                suffix='.tmp', delete=False) as f:
            json.dump(self._output, f, indent=self._indent)
            f.flush()
            os.fsync(f)
            f.close()
            os.replace(f.name, self._output_file)

    def append_all(self, top_level_key, iteration, metrics):
        self.append("%s/iteration" % top_level_key, iteration)
        self.append("%s/time" % top_level_key, time())
        for metric, value in metrics.items():
            self.append("%s/%s" % (top_level_key, metric), value)
