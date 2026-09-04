from typing import Dict, List, Optional
from app.tools.base import BaseTool
from app.tools.implementations import (
    VQATool,
    CaptioningTool,
    GroundingTool,
    ChangeDetectionTool,
    OpticalSARTool,
    MetadataReaderTool,
    EvidenceFusionTool,
    ReportGeneratorTool
)

class ToolRegistry:
    """
    Central Tool Registry maintaining available models and tools.
    Allows dynamic lookup, registration, listing, and execution.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        # Register standard default tools
        self.register(VQATool())
        self.register(CaptioningTool())
        self.register(GroundingTool())
        self.register(ChangeDetectionTool())
        self.register(OpticalSARTool())
        self.register(MetadataReaderTool())
        self.register(EvidenceFusionTool())
        self.register(ReportGeneratorTool())

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools and descriptions."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def execute_tool(self, name: str, payload: Dict) -> Dict:
        """Execute a tool by name with safety checks."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' is not registered in the ToolRegistry.")
        missing = [key for key in tool.required_inputs if key not in (payload or {})]
        if missing:
            raise ValueError(f"Tool '{name}' missing required inputs: {missing}")
        return tool.execute(payload or {})

tool_registry = ToolRegistry()
