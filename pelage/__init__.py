# ruff: noqa: F403
import importlib.metadata

__version__ = importlib.metadata.version("pelage")

from .checks import *
