import os
import pathlib

from setuptools import find_packages, setup


def get_version() -> str:
    init = open(os.path.join("flowrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()
VERSION = get_version()

def get_install_requires():
    return [
        "jax[cuda12]==0.5.3",
        "flax==0.10.5",
        "gymnasium",
        'shimmy==1.3.0',
        "gym==0.23.1",
        "mujoco_py==2.1.2.14",
        "dm_control<=1.0.20", # https://github.com/Farama-Foundation/D4RL/issues/236
        "mujoco<=3.1.6", # https://github.com/Farama-Foundation/D4RL/issues/236
        'Cython<3',
        'six==1.17.0',
        "tqdm",
        "hydra-core",
        "distrax",
        "tensorboardX==2.6.2.2",
        "scikit-learn==1.6.1",
        "wandb",
    ]

def get_extras_require():
    return {}

setup(
    name                = "flowrl",
    version             = VERSION,
    description         = "A library desgined for flow-based RL algorithms",
    long_description    = README,
    long_description_content_type = "text/markdown",
    url                 = "https://github.com/typoverflow/flow-rl",
    author              = "typoverflow",
    author_email        = "typoverflow@gmail.com",
    license             = "MIT",
    packages            = find_packages(),
    include_package_data = True,
    tests_require=["pytest", "mock"],
    python_requires=">=3.11",
    install_requires = get_install_requires()
)
