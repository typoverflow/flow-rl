import os
import pathlib

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


def get_version() -> str:
    init = open(os.path.join("flowrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()
VERSION = get_version()

def get_base_requirements():
    return [
        "jax[cuda12]==0.5.3",
        "flax==0.10.5",
        "orbax-checkpoint==0.11.23",
        "gymnasium",
        'shimmy==1.3.0',
        'Cython<3',
        'six==1.17.0',
        "tqdm",
        "hydra-core",
        "distrax",
        "tensorboardX==2.6.2.2",
        "scikit-learn==1.6.1",
        "wandb",
        "matplotlib",
    ]

def get_install_requires():
    return get_base_requirements()

def get_extras_require():
    return {
        'mujoco_offline': [
            "dm_control<=1.0.20",
            "mujoco<=3.1.6",
        ],
        'mujoco_online': [
            "dm_control==1.0.27",
            "mujoco==3.2.7",
        ],
        'humanoidbench': [
            "torch",
            "stable-baselines3==2.3.2",
            "mujoco<=3.1.6",
            "humanoid-bench @ git+https://github.com/carlosferrazza/humanoid-bench.git",
        ],
        'legacy_mujoco': [
            "mujoco_py==2.1.2.14",
        ],
        'robotics': [
            "gymnasium_robotics",
        ],
    }

def get_ext_modules():
    return [
        Pybind11Extension(
            "data_structure",
            ["flowrl/data_structure/data_structure.cc"],
            define_macros=[("VERSION_INFO", "\"{}\"".format(VERSION))],
        )
    ]

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
    ext_modules = get_ext_modules(),
    cmdclass={"build_ext": build_ext},
    install_requires = get_install_requires(),
    extras_require = get_extras_require(),
)
