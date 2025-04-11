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
        "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl",
        "git+https://github.com/aravindr93/mjrl@master#egg=mjrl", # for d4rl
        "gymnasium==0.29.1", # We can't use gymnasium >= 1.0.0 because it breaks d4rl.
        "gym==0.23.1", # for d4rl
        "numpy==1.26.4", # for d4rl
        "shimmy[gym-v21,gym-v26]==1.0.0", # for d4rl
        'cython<3' # for d4rl
        'six==1.17.0', # for mjrl
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
    python_requires=">=3.12",
    install_requires = get_install_requires()
)