"""Compatibility imports for the former model module path."""
from .model import FACA as PaperFACA
from .model import FACAConfig as PaperConfig
from .model import MeasuredState, TaskDescriptorEncoder

__all__ = ['PaperFACA', 'PaperConfig', 'MeasuredState', 'TaskDescriptorEncoder']
