"""
src/agent/orchestrator.py

VoiceAgent: a ReAct agent that accepts text or audio input,
reasons using an LLM, calls tools, and responds via TTS in the user's language.

ReAct loop (Yao et al. 2022):
    Thought → Action (tool call) → Observation → repeat → Final Answer

The agent detects the user's language (via Whisper for audio, langdetect for text)
and passes the language code to the TTS router so the spoken response
is always in the same language the user used.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from src.agent.tools import call_tool, tools_prompt
from src.tts.router import TTSRouter
from src.memory.vector_store import VectorMemory
from src.utils.lang_detect import detect_language


SYSTEM_PROMPT_TEMPLATE = """You are a helpful multilingual voice assistant.
Respond in the same language the user used. Be concise — your answer will be spoken aloud.

{tools}

Use this format strictly:
Thought: <reason about what to do>
Action: <tool_name>
Action Input: <input to the tool>
Observation: <tool result — filled in by the system>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: <your answer in the user's language>
"""

FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(.+?)(?=\nObservation|\nThought|$)", re.DOTALL)


@dataclass
class Turn:
    user_text: str
    language: str
    agent_text: str
    audio_bytes: bytes | None = None


class VoiceAgent:
    """
    ReAct agent with TTS output and episodic memory.

    Args:
        llm_fn: Callable(messages: list[dict]) -> str. Any LLM backend.
        tts_router: TTSRouter instance.
        memory: VectorMemory instance.
        max_steps: Max ReAct iterations before giving up.
    """

    def __init__(
        self,
        llm_fn,
        tts_router: TTSRouter,
        memory: VectorMemory,
        max_steps: int = 5,
    ):
        self.llm = llm_fn
        self.tts = tts_router
        self.memory = memory
        self.max_steps = max_steps

    def run(self, user_text: str, lang: str | None = None) -> Turn:
        """
        Process a text input and return a Turn with text + audio response.

        Args:
            user_text: The user's input text.
            lang: ISO 639-1 code. If None, auto-detected from user_text.
        """
        if lang is None:
            lang = detect_language(user_text)

        # Retrieve relevant past context
        past = self.memory.retrieve(user_text, k=3)
        context_str = "\n---\n".join(past) if past else "No prior context."

        system = SYSTEM_PROMPT_TEMPLATE.format(tools=tools_prompt())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Past context:\n{context_str}\n\n"
                f"User ({lang}): {user_text}"
            )},
        ]

        scratchpad = ""
        final_answer = None

        for step in range(self.max_steps):
            response = self.llm(messages + [{"role": "assistant", "content": scratchpad}])
            scratchpad += response

            # Check for final answer
            match = FINAL_ANSWER_RE.search(scratchpad)
            if match:
                final_answer = match.group(1).strip()
                break

            # Check for tool call
            action_match = ACTION_RE.search(response)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                observation = call_tool(tool_name, tool_input)
                scratchpad += f"\nObservation: {observation}\n"
            else:
                # No action, no final answer — nudge the model
                scratchpad += "\nThought: I should provide a Final Answer now.\n"

        if final_answer is None:
            final_answer = "I'm sorry, I wasn't able to find an answer."

        # Store this turn in episodic memory
        self.memory.store(user_text, final_answer, lang=lang)

        # Synthesize audio
        audio_bytes = self.tts.synthesize(final_answer, lang=lang)

        return Turn(
            user_text=user_text,
            language=lang,
            agent_text=final_answer,
            audio_bytes=audio_bytes,
        )
