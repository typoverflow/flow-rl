import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np


def make_unique_name(name):
    name = name or ""
    now = datetime.now()
    suffix = now.strftime("%m-%d-%H-%M")
    pid_str = os.getpid()
    if name == "":
        return f"{suffix}-{pid_str}"
    else:
        return f"{name}-{suffix}-{pid_str}"

def fmt_time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

cmap = {
    None: "\033[0m",
    "error": "\033[1;31m",
    "debug": "\033[0m",
    "warning": "\033[1;33m",
    "info": "\033[1;34m",
    "reset": "\033[0m",
}

def log(msg: str, type: str):
    time_str = fmt_time_now()
    print("{}[{}]{}\t{}".format(
        cmap.get(type.lower(), "\033[0m"),
        time_str,
        cmap["reset"],
        msg
    ))


class LogLevel:
    NOTSET = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    INFO = 4


class BaseLogger():
    """
    Base class for loggers, providing basic string logging.
    """

    cmap = {
        None: "\033[0m",
        "error": "\033[1;31m",
        "debug": "\033[0m",
        "warning": "\033[1;33m",
        "info": "\033[1;34m",
        "reset": "\033[0m",
    }

    def __init__(
        self,
        log_dir: str,
        name: Optional[str]=None,
        unique_name: Optional[str]=None,
        backup_stdout: bool=False,
        activate: bool=True,
        level: int=LogLevel.WARNING,
        *args, **kwargs
    ):
        self.activate = activate
        if not self.activate:
            return
        if unique_name:
            self.unique_name = unique_name
        else:
            self.unique_name = make_unique_name(name)
        self.log_dir = os.path.join(log_dir, self.unique_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.backup_stdout = backup_stdout
        if self.backup_stdout:
            self.stdout_file = os.path.join(self.log_dir, "stdout.txt")
            self.stdout_fp = open(self.stdout_file, "w+")
        self.output_dir = os.path.join(self.log_dir, "output")
        self.level = level

    def can_log(self, level=LogLevel.INFO):
        return self.activate and level >= self.level

    def _write(self, time_str: str, msg: str, type="info"):
        type = type.upper()
        self.stdout_fp.write("[{}] ({})\t{}\n".format(
            time_str,
            type,
            msg
        ))
        self.stdout_fp.flush()

    def info(self, msg: str, level: int=LogLevel.INFO):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["info"],
                time_str,
                self.cmap["reset"],
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "info")


    def debug(self, msg: str, level: int=LogLevel.DEBUG):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["debug"],
                time_str,
                self.cmap["reset"],
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "debug")

    def warning(self, msg: str, level: int=LogLevel.WARNING):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["warning"],
                time_str,
                self.cmap["reset"],
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "warning")

    def error(self, msg: str, level: int=LogLevel.ERROR):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["error"],
                time_str,
                self.cmap["reset"],
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "error")

    def log_str(self, msg: str, type: Optional[str]=None, *args, **kwargs):
        if type: type = type.lower()
        level = {
            None: LogLevel.DEBUG,
            "error": LogLevel.ERROR,
            "log": LogLevel.INFO,
            "warning": LogLevel.WARNING,
            "debug": LogLevel.DEBUG
        }.get(type)
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap[type],
                time_str,
                self.cmap["reset"],
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, type)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "stdout_fp"):
            self.stdout_fp.close()

class TensorboardLogger(BaseLogger):
    """
    Tensorboard Logger

    Parameters
    ----------
    log_dir :  The base dir where the logger logs to.
    name :  The name of the experiment, will be used to construct the event file name. A suffix
            will be added to the name to ensure the uniqueness of the log dir.
    unique_name :  The name of the experiment, but no suffix will be appended.
    backup_stdout :  Whether or not backup stdout to files.
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message.
    """

    def __init__(
        self,
        log_dir: str,
        name: Optional[str]=None,
        unique_name: Optional[str]=None,
        backup_stdout: bool=False,
        activate: bool=True,
        level=LogLevel.WARNING,
        *args, **kwargs
    ):
        super().__init__(log_dir, name, unique_name, backup_stdout, activate, level)
        if not self.activate:
            return
        from tensorboardX import SummaryWriter
        self.tb_dir = os.path.join(self.log_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        # self.output_dir = self.tb_dir
        self.tb_writer = SummaryWriter(self.tb_dir)

    def log_scalar(
        self,
        tag: str,
        value: Union[float, int],
        step: Optional[int]=None
    ):
        """Add scalar to tensorboard summary.

        tag :  the identifier of the scalar.
        value :  value to record.
        step :  global timestep of the scalar.
        """
        if not self.can_log():
            return
        self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int]],
        step: Optional[int]=None
    ):
        """Add scalars which share the main tag to tensorboard summary.

        main_tag :  the shared main tag of the scalars, can be a null string.
        tag_scalar_dict :  a dictionary of tag and value.
        step :  global timestep of the scalars.
        """
        if not self.can_log():
            return
        if main_tag is None or main_tag == "":
            main_tag = ""
        else:
            main_tag = main_tag+"/"

        for tag, value in tag_scalar_dict.items():
            self.tb_writer.add_scalar(main_tag+tag, value, step)

    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        step: Optional[int]=None,
        dataformat: str="CHW"
    ):
        """Add image to tensorboard summary. Note that this requires ``pillow`` package.

        :param tag: the identifier of the image.
        :param img_tensor: an `uint8` or `float` Tensor of shape `
                [channel, height, width]` where `channel` is 1, 3, or 4.
                The elements in img_tensor can either have values
                in [0, 1] (float32) or [0, 255] (uint8).
                Users are responsible to scale the data in the correct range/type.
        :param global_step: global step.
        :param dataformats: This parameter specifies the meaning of each dimension of the input tensor.
        """
        if not self.can_log():
            return
        self.tb_writer.add_image(tag, img_tensor, step, dataformats=dataformat)

    def log_video(
        self,
        tag: str,
        vid_tensor: Any,
        step: Optional[int]=None,
        fps: Optional[Union[float, int]]=4,
    ):
        """Add a piece of video to tensorboard summary. Note that this requires ``moviepy`` package.

        :param tag: the identifier of the video.
        :param vid_tensor: video data.
        :param global_step: global step.
        :param fps: frames per second.
        :param dataformat: specify different permutation of the video tensor.
        """
        if not self.can_log():
            return
        self.tb_writer.add_video(tag, vid_tensor, step, fps)

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, List],
        step: Optional[int]=None,
    ):
        """Add histogram to tensorboard.

        :param tag: the identifier of the histogram.
        :param values: the values, should be list or np.ndarray.
        :param global_step: global step.
        """
        if not self.can_log():
            return
        self.tb_writer.add_histogram(tag, np.asarray(values), step)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tb_writer.close()
