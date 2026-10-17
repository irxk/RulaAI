"""Dedicated RulaCoder agent configuration for the shared local 9B engine."""
from .agent import Agent
from .coder_workflow import CoderWorkflow
from .tools import ToolRegistry


class CoderAgent(Agent):
    """Keep coding policy and request budgets separate from Rula's chat agent."""

    def configure_request(self, mode: str, effort: str, tools: ToolRegistry) -> None:
        workflow = CoderWorkflow(mode, effort)
        budget = workflow.budget
        self.tools = tools
        self.config.extra_system = workflow.system_prompt()
        self.config.coder_auto = mode == "auto"
        self.config.max_tool_calls = (
            budget["tool_calls"] if mode == "auto"
            else min(budget["tool_calls"], 8))
        self.config.max_new_tokens = budget["tokens"]
        self.config.max_continuations = budget["continuations"]
        self.config.temperature = 0.2
        self.config.stream = False
        self.config.preview = False
        self.config.thinking = CoderWorkflow.thinking_enabled(effort, self.engine)
