"""
agent/orchestrator.py

Minimal ReAct agent loop.

Thought → Action (tool call) → Observation → repeat → Final Answer

The agent detects the user's language and passes it to the TTS router
so the response is spoken in the same language the user used.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.tts.router import TTSRouter
from src.memory.vector_store import VectorMemory
from src.utils.lang_detect import detect_language
from src.agent.tools import TOOLS


@dataclass
class AgentState:
    history: list[dict] = field(default_factory=list)
    detected_lang: str = "en"


class VoiceAgent:
    def __init__(self, llm, tts_router: TTSRouter, memory: VectorMemory):
        self.llm = llm
        self.tts = tts_router
        self.memory = memory

    def run(self, user_input: str, max_steps: int = 5) -> bytes:
        """
        Process user_input, run the ReAct loop, and return audio bytes.
        """
        state = AgentState()
        state.detected_lang = detect_language(user_input)

        # Retrieve relevant memory
        past_context = self.memory.retrieve(user_input, k=3)

        # Build initial prompt
        messages = self._build_messages(user_input, past_context)

        for step in range(max_steps):
            response = self.llm(messages)

            if response.is_final_answer:
                # Store turn in memory
                self.memory.store(user_input, response.text)
                # Speak the answer
                return self.tts.synthesize(response.text, lang=state.detected_lang)

            # Execute tool
            tool_result = self._call_tool(response.action, response.action_input)
            messages.append({"role": "tool", "content": tool_result})

        raise RuntimeError("Agent exceeded max_steps without a final answer")

    def _call_tool(self, tool_name: str, tool_input: str) -> str:
        tool = TOOLS.get(tool_name)
        if tool is None:
            return f"Error: tool '{tool_name}' not found"
        return tool(tool_input)

    def _build_messages(self, user_input: str, past_context: list[str]) -> list[dict]:
        system = (
            "You are a helpful multilingual voice assistant. "
            "Use tools when needed. Always respond in the same language as the user. "
            "Available tools: web_search, retrieve_memory, detect_language, speak."
        )
        context_str = "\n".join(past_context) if past_context else "No relevant past context."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Past context:\n{context_str}\n\nUser: {user_input}"},
        ]
