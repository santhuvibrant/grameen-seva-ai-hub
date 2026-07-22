"""Two-phase Gemini conversation agent: collect first, research second."""

from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from google.genai import types

from agents.prompts import EXTRACTION_PROMPT, RESEARCH_PROMPT
from models.conversation import AgentResult, ConversationState
from services.research import get_scheme_details, search_schemes


def _parse(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"voice_response": cleaned[:500], "next_question": "", "conversation_complete": False}


def _history(state: ConversationState) -> str:
    return "\n".join(f"{turn['role']}: {turn['text']}" for turn in state.turns)


def _tool_declarations() -> list[types.Tool]:
    return [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="search_schemes",
                description="Search official Indian government agricultural schemes.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(type="STRING"),
                        "state": types.Schema(type="STRING"),
                    },
                    required=["query", "state"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_scheme_details",
                description="Read an official scheme page.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"url": types.Schema(type="STRING")},
                    required=["url"],
                ),
            ),
        ])
    ]


def _run_tool(name: str, args: dict[str, Any], state: ConversationState, keys: dict[str, str]) -> str:
    if name == "search_schemes":
        return search_schemes(args.get("query", ""), args.get("state", ""), keys["tavily"])
    if name == "get_scheme_details":
        return get_scheme_details(args.get("url", ""), keys["firecrawl"])
    return json.dumps({"error": "Unknown tool"})


def _generate(client: Any, prompt: str, system: str, tools: list[types.Tool] | None = None) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(system_instruction=system, tools=tools or [], temperature=0.2),
    )
    return response.text or ""


def run_conversation(state: ConversationState, gemini_key: str, tavily_key: str, firecrawl_key: str) -> AgentResult:
    client = genai.Client(api_key=gemini_key)
    detected = state.language_code or "unknown"
    extraction = _parse(_generate(
        client,
        f"Detected language: {detected}\nConversation:\n{_history(state)}",
        EXTRACTION_PROMPT,
    ))
    preliminary = AgentResult.from_dict(extraction, detected)
    if not preliminary.conversation_complete:
        return preliminary

    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=(
        f"Detected language: {detected}\nCollected farmer information:\n{_history(state)}\n"
        "Research the best matching official scheme and return the required JSON."
    ))])]
    for _ in range(8):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=RESEARCH_PROMPT, tools=_tool_declarations(), temperature=0.2),
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts if candidate.content else []
        calls = [part.function_call for part in parts if part.function_call]
        if not calls:
            result = AgentResult.from_dict(_parse(response.text or ""), detected)
            result.conversation_complete = True
            return result
        contents.append(candidate.content)
        contents.append(types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            name=call.name,
            response={"result": _run_tool(call.name, dict(call.args or {}), state, {"tavily": tavily_key, "firecrawl": firecrawl_key})},
        )) for call in calls]))
    return AgentResult(language=detected, conversation_complete=True, voice_response="I could not complete the official source lookup.")
