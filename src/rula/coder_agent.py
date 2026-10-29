"""Dedicated RulaCoder configuration and optional structured planning."""
import os

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
        self.config.coder_auto = mode in ("generate", "auto")
        self.config.max_tool_calls = (
            budget["tool_calls"] if mode in ("generate", "auto")
            else min(budget["tool_calls"], 8))
        self.config.max_new_tokens = budget["tokens"]
        self.config.max_continuations = budget["continuations"]
        self.config.temperature = 0.2
        self.config.stream = False
        self.config.preview = False
        planner_context = ""
        if mode in ("generate", "auto") and getattr(
                self.engine, "backend", None) in {"tabbyapi", "exllamav2"}:
            if getattr(self.engine, "backend", None) == "exllamav2":
                self.engine.close()
            progress = getattr(self, "on_progress", None)
            if callable(progress):
                progress("Llama Plannerで実装計画を作成しています")
            from .coder_planner import create_coding_plan, format_coding_plan
            from .engine import load_env_file
            from .tabbyapi_backend import TabbyAPIEngine

            load_env_file()
            planner_model = (
                os.environ.get("RULA_CODER_PLANNER_MODEL", "").strip()
                or os.environ.get(
                    "RULA_TABBYAPI_MODEL", "Llama-3.1-8B-Instruct-exl3"
                ).strip()
            )
            planner = TabbyAPIEngine(
                model=planner_model,
                model_env="RULA_CODER_PLANNER_MODEL",
                display_name="Llama 3.1 8B Planner",
            )
            try:
                plan = create_coding_plan(planner, request_text, memory_context)
            finally:
                planner.close()
            planner_context = format_coding_plan(plan)
        sections = [prompt]
        if planner_context:
            sections.append(planner_context)
        if memory_context:
            sections.append(memory_context)
        self.config.extra_system = "\n\n".join(sections)
        action_request = mode in ("generate", "auto") and CoderWorkflow.is_action_request(
            request_text)
        self.config.thinking = (
            CoderWorkflow.thinking_enabled(effort, self.engine) and not action_request)
