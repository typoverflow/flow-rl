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
    print(
        "{}[{}]{}\t{}".format(
            cmap.get(type.lower(), "\033[0m"), time_str, cmap["reset"], msg
        )
    )


class LogLevel:
    NOTSET = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    INFO = 4


class BaseLogger:
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
        name: Optional[str] = None,
        unique_name: Optional[str] = None,
        backup_stdout: bool = False,
        activate: bool = True,
        level: int = LogLevel.WARNING,
        *args,
        **kwargs,
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
        self.stdout_fp.write("[{}] ({})\t{}\n".format(time_str, type, msg))
        self.stdout_fp.flush()

    def info(self, msg: str, level: int = LogLevel.INFO):
        if self.can_log(level):
            time_str = fmt_time_now()
            print(
                "{}[{}]{}\t{}".format(
                    self.cmap["info"], time_str, self.cmap["reset"], msg
                )
            )
            if self.backup_stdout:
                self._write(time_str, msg, "info")

    def debug(self, msg: str, level: int = LogLevel.DEBUG):
        if self.can_log(level):
            time_str = fmt_time_now()
            print(
                "{}[{}]{}\t{}".format(
                    self.cmap["debug"], time_str, self.cmap["reset"], msg
                )
            )
            if self.backup_stdout:
                self._write(time_str, msg, "debug")

    def warning(self, msg: str, level: int = LogLevel.WARNING):
        if self.can_log(level):
            time_str = fmt_time_now()
            print(
                "{}[{}]{}\t{}".format(
                    self.cmap["warning"], time_str, self.cmap["reset"], msg
                )
            )
            if self.backup_stdout:
                self._write(time_str, msg, "warning")

    def error(self, msg: str, level: int = LogLevel.ERROR):
        if self.can_log(level):
            time_str = fmt_time_now()
            print(
                "{}[{}]{}\t{}".format(
                    self.cmap["error"], time_str, self.cmap["reset"], msg
                )
            )
            if self.backup_stdout:
                self._write(time_str, msg, "error")

    def log_str(self, msg: str, type: Optional[str] = None, *args, **kwargs):
        if type:
            type = type.lower()
        level = {
            None: LogLevel.DEBUG,
            "error": LogLevel.ERROR,
            "log": LogLevel.INFO,
            "warning": LogLevel.WARNING,
            "debug": LogLevel.DEBUG,
        }.get(type)
        if self.can_log(level):
            time_str = fmt_time_now()
            print(
                "{}[{}]{}\t{}".format(
                    self.cmap[type], time_str, self.cmap["reset"], msg
                )
            )
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
        name: Optional[str] = None,
        unique_name: Optional[str] = None,
        backup_stdout: bool = False,
        activate: bool = True,
        level=LogLevel.WARNING,
        *args,
        **kwargs,
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
        self, tag: str, value: Union[float, int], step: Optional[int] = None
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
        step: Optional[int] = None,
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
            main_tag = main_tag + "/"

        for tag, value in tag_scalar_dict.items():
            self.tb_writer.add_scalar(main_tag + tag, value, step)

    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        step: Optional[int] = None,
        dataformat: str = "CHW",
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
        step: Optional[int] = None,
        fps: Optional[Union[float, int]] = 4,
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
        step: Optional[int] = None,
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


class CsvLogger(BaseLogger):
    """
    CSV Logger

    Parameters
    ----------
    log_dir :  The base dir where the logger logs to.
    name :  The name of the experiment, will be used to construct the event file name. A suffix
            will be added to the name to ensure the uniqueness of the log path.
    unique_name :  The name of the experiment, but no suffix will be appended.
    backup_stdout :  Whether or not backup stdout to files.
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message.
    """

    def __init__(
        self,
        log_dir: str,
        name: Optional[str] = None,
        unique_name: Optional[str] = None,
        backup_stdout: bool = False,
        activate: bool = True,
        level=LogLevel.WARNING,
        *args,
        **kwargs,
    ):
        super().__init__(log_dir, name, unique_name, backup_stdout, activate, level)
        if not self.activate:
            return
        self.csv_dir = os.path.join(self.log_dir, "csv")
        self.csv_file = os.path.join(self.csv_dir, "output.csv")
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        self.csv_fp = open(self.csv_file, "w+")
        self.csv_sep = ","
        self.csv_keys = ["step"]

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ):
        """Add scalar to CSV file.

        tag :  the identifier of the scalar.
        value :  value to record.
        step :  global timestep of the scalar.
        """
        if not self.can_log():
            return
        if main_tag is None or main_tag == "":
            pass
        else:
            main_tag = main_tag + "/"
            tag_scalar_dict = {
                main_tag + tag: value for tag, value in tag_scalar_dict.items()
            }
        # handle new keys
        extra_keys = list(tag_scalar_dict.keys() - self.csv_keys)
        extra_keys.sort()
        if extra_keys:
            self.csv_keys.extend(extra_keys)
            self.csv_fp.seek(0)
            lines = self.csv_fp.readlines()
            self.csv_fp = open(self.csv_file, "w+t")
            self.csv_fp.seek(0)
            self.csv_fp.write(",".join(self.csv_keys) + "\n")
            for line in lines[1:]:
                self.csv_fp.write(line[:-1])
                self.csv_fp.write(self.csv_sep * len(extra_keys))
                self.csv_fp.write("\n")
            self.csv_fp = open(self.csv_file, "a+t")
        # write new entry
        values_to_write = [
            str(tag_scalar_dict.get(key, "")) if key != "step" else str(int(step))
            for key in self.csv_keys
        ]
        self.csv_fp.write(",".join(values_to_write) + "\n")
        self.csv_fp.flush()

    def log_scalar(
        self, tag: str, value: Union[float, int], step: Optional[int] = None
    ):
        """Add scalar to CSV summary.

        tag :  the identifier of the scalar.
        value :  value to record.
        step :  global timestep of the scalar.
        """
        self.log_scalars(main_tag=None, tag_scalar_dict={tag: value}, step=step)

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exc_val, exc_tb):
        self.csv_fp.close()


class WandbLogger(BaseLogger):
    """
    WandB Logger

    Parameters
    ----------
    log_dir :  The base dir where the logger logs to.
    name :  The name of the experiment, will be used to construct the event file name. A suffix
            will be added to the name to ensure the uniqueness of the log dir.
    config :  The the configs or hyper-parameters of the experiment, should be dict-like.
    project :  The project for wandb.
    entity :  The entity for wandb.
    unique_name :  The name of the experiment, but no suffix will be appended.
    backup_stdout :  Whether or not backup stdout to files.
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message.
    """

    def __init__(
        self,
        log_dir: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        unique_name: Optional[str] = None,
        backup_stdout: bool = False,
        activate: bool = True,
        level: int = LogLevel.WARNING,
        *args,
        **kwargs,
    ):
        super().__init__(log_dir, name, unique_name, backup_stdout, activate, level)
        if not self.activate:
            return
        import wandb

        self.run = wandb.init(
            dir=self.log_dir,
            name=self.unique_name,
            config=config,
            project=project,
            entity=entity,
            **kwargs,
        )  # this create the `self.log_dir/wandb` dir
        # define the custom timestep metric
        self.run.define_metric("step")
        self.keys = set()

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ):
        if not self.can_log():
            return
        if main_tag is None or main_tag == "":
            pass
        else:
            main_tag = main_tag + "/"
            tag_scalar_dict = {
                main_tag + key: value for key, value in tag_scalar_dict.items()
            }
        # handle new keys
        extra_keys = set(tag_scalar_dict.keys()).difference(self.keys)
        for ek in extra_keys:
            self.run.define_metric(ek, step_metric="step")
        self.keys = self.keys.union(extra_keys)
        tag_scalar_dict["step"] = step
        self.run.log(tag_scalar_dict, step=None)

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        self.log_scalars(main_tag=None, tag_scalar_dict={tag: value}, step=step)

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exc_val, exc_tb):
        self.run.finish()


class CompositeLogger(BaseLogger):
    """
    Composite Logger, which composes multiple logger implementations to provide a unified
    interface for various types of log recording.

    Parameters
    ----------
    log_dir :  The base dir where the logger logs to.
    name :  The name of the experiment, will be used to construct the event file name. A suffix
            will be added to the name to ensure the uniqueness of the log dir.
    logger_config :  The ad-hoc configs for each component logger.
    unique_name :  The name of the experiment, but no suffix will be appended.
    backup_stdout :  Whether or not backup stdout to files.
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message.
    """

    logger_registry = {
        "CsvLogger": CsvLogger,
        "TensorboardLogger": TensorboardLogger,
        "WandbLogger": WandbLogger,
    }

    def __init__(
        self,
        log_dir: str,
        name: Optional[str] = None,
        logger_config: Dict = {},
        unique_name: Optional[str] = None,
        backup_stdout: bool = False,
        activate: bool = True,
        level: int = LogLevel.WARNING,
        *args,
        **kwargs,
    ):
        super().__init__(log_dir, name, unique_name, backup_stdout, activate, level)
        if not self.activate:
            return
        # create loggers
        default_config = {
            "log_dir": log_dir,
            # "name": name,
            "unique_name": self.unique_name,
            "backup_stdout": False,
            "activate": self.activate,
            "level": self.level,
        }
        self.loggers = dict()
        self.logger_config = dict()
        self.logger_cls = set()
        for logger_cls in logger_config:
            config = default_config.copy()
            config.update(logger_config[logger_cls])
            config["backup_stdout"] = (
                False  # force sub loggers not to backup to avoid multiple file handles
            )
            self.logger_config[logger_cls] = config
            # print(logger_cls, config)
            if config.get("activate", True) == False:
                continue
            self.loggers[logger_cls] = self.logger_registry[logger_cls](**config)
            self.logger_cls.add(logger_cls)

    def _try_call_by_group(self, func: str, group: list, *args, **kwargs):
        if self.can_log():
            return {
                _logger_cls: getattr(self.loggers[_logger_cls], func)(*args, **kwargs)
                for _logger_cls in group
                if _logger_cls in self.logger_cls
            }

    def log_scalar(
        self, tag: str, value: Union[float, int], step: Optional[int] = None
    ):
        return self._try_call_by_group(
            func="log_scalar",
            group=["TensorboardLogger", "WandbLogger", "CsvLogger"],
            tag=tag,
            value=value,
            step=step,
        )

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ):
        return self._try_call_by_group(
            func="log_scalars",
            group=["TensorboardLogger", "WandbLogger", "CsvLogger"],
            main_tag=main_tag,
            tag_scalar_dict=tag_scalar_dict,
            step=step,
        )

    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        step: Optional[int] = None,
        dataformat: str = "CHW",
    ):
        return self._try_call_by_group(
            func="log_image",
            group=["TensorboardLogger"],
            tag=tag,
            img_tensor=img_tensor,
            step=step,
            dataformat=dataformat,
        )

    def log_video(
        self,
        tag: str,
        vid_tensor: Any,
        step: Optional[int] = None,
        fps: Optional[Union[float, int]] = 4,
    ):
        return self._try_call_by_group(
            func="log_video",
            group=["TensorboardLogger"],
            tag=tag,
            vid_tensor=vid_tensor,
            step=step,
            fps=fps,
        )

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, List],
        step: Optional[int] = None,
    ):
        return self._try_call_by_group(
            func="log_histogram",
            group=["TensorboardLogger"],
            tag=tag,
            values=values,
            step=step,
        )

    def log_object(
        self,
        name: str,
        object: Any,
        path: Optional[str] = None,
        protocol: str = "torch",
    ):
        return self._try_call_by_group(
            func="log_object",
            group=["TensorboardLogger"],
            name=name,
            object=object,
            path=path,
            protocol=protocol,
        )
