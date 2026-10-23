"""Dedicated RulaCoder agent configuration for the shared local 9B engine."""
from .agent import Agent
from .coder_workflow import CoderWorkflow
from .tools import ToolRegistry


class CoderAgent(Agent):
    """Keep coding policy and request budgets separate from Rula's chat agent."""

    def configure_request(
            self, mode: str, effort: str, tools: ToolRegistry,
            request_text: str = "", memory_context: str = "") -> None:
        workflow = CoderWorkflow(mode, effort)
        budget = workflow.budget
        self.tools = tools
        prompt = workflow.system_prompt(request_text)
        self.config.extra_system = (
            f"{prompt}\n\n{memory_context}" if memory_context else prompt)
        self.config.coder_auto = mode in ("generate", "auto")
        self.config.max_tool_calls = (
            budget["tool_calls"] if mode in ("generate", "auto")
            else min(budget["tool_calls"], 8))
        self.config.max_new_tokens = budget["tokens"]
        self.config.max_continuations = budget["continuations"]
        self.config.temperature = 0.2
        self.config.stream = False
        self.config.preview = False
        action_request = mode in ("generate", "auto") and CoderWorkflow.is_action_request(
            request_text)
        self.config.thinking = (
            CoderWorkflow.thinking_enabled(effort, self.engine) and not action_request)
