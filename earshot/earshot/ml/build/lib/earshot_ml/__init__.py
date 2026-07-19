"""Earshot ML: offline sound-event detection on a Pi, plus teach mode."""

from . import config
from .core import EarshotML, Event

__all__ = ["EarshotML", "Event", "config"]
