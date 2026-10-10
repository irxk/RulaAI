from .base import Tool, ToolRegistry
from .calc import CALC_TOOL
from .dates import DATE_TOOL
from .search import (DEEP_SEARCH_TOOL, READ_URL_TOOL, SEARCH_TOOL,
                     SMART_SEARCH_TOOL)


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (SEARCH_TOOL, DEEP_SEARCH_TOOL, SMART_SEARCH_TOOL,
                 READ_URL_TOOL, CALC_TOOL, DATE_TOOL):
        reg.register(tool)
    return reg


__all__ = ["Tool", "ToolRegistry", "default_registry",
           "SEARCH_TOOL", "DEEP_SEARCH_TOOL", "SMART_SEARCH_TOOL",
           "READ_URL_TOOL", "CALC_TOOL", "DATE_TOOL"]
