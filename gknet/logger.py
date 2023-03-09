# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _ensure_dir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


# custom json serializer for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Logger:
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        _ensure_dir(opt.save_dir)
        _ensure_dir(opt.debug_dir)

        time_str = time.strftime("%Y-%m-%d-%H-%M")

        args = dict(
            (name, getattr(opt, name)) for name in dir(opt) if not name.startswith("_")
        )
        file_name = os.path.join(opt.save_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("==> torch version: {}\n".format(torch.__version__))
            opt_file.write(
                "==> cudnn version: {}\n".format(torch.backends.cudnn.version())
            )
            opt_file.write("==> Cmd:\n")
            opt_file.write(str(sys.argv))
            opt_file.write("\n==> Opt:\n")
            opt_file.write(json.dumps(args, indent=2, sort_keys=True, cls=NumpyEncoder))

        log_dir = f"{opt.save_dir}/logs_{time_str}"
        _ensure_dir(log_dir)
        shutil.copyfile(f"{opt.save_dir}/opt.txt", f"{log_dir}/opt.txt")

        self.log = open(log_dir + "/log.txt", "w")
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime("%Y-%m-%d-%H-%M")
            self.log.write("{}: {}".format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if "\n" in txt:
            self.start_line = True
            self.log.flush()

    def write_line(self, txt):
        self.write(txt + "\n")

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        pass
