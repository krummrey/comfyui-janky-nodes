
# __init__.py
from .informative_drawing_node import InformativeDrawingNode
from .local_contrast_node import LocalContrastNode
from .smoothing_node import SmoothingNode

# Node Class Mappings
NODE_CLASS_MAPPINGS = {
    "InformativeDrawingNode": InformativeDrawingNode,
    "LocalContrastNode": LocalContrastNode,
    "SmoothingNode": SmoothingNode,
}

# Node Display Name Mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "InformativeDrawingNode": "Informative Drawing",
    "LocalContrastNode": "Local Contrast Enhancement",
    "SmoothingNode": "Edge-Aware Smoothing",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"