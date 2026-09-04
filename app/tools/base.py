from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseTool(ABC):
    """
    Abstract Base Class for all tools registered in the Tool Registry.
    Each tool wraps an underlying model adapter (e.g. CD Chat, GeoChat, Popeye, ResNet)
    or utility pipeline and exposes a unified execute interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description."""
        pass

    @property
    @abstractmethod
    def required_inputs(self) -> List[str]:
        """List of required payload parameter keys."""
        pass

    @abstractmethod
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool action given the payload.
        Returns standard dictionary with model execution outputs and confidence score.
        """
        pass
