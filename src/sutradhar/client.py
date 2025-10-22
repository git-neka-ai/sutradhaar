# orion: Isolated the OpenAI client wrapper (Responses and Chat Completions compatibility) and HTTP dump helper to keep API concerns separate from Orion logic.

import json
from typing import Any, Dict, List, Optional

import requests

from .config import MAX_COMPLETION_TOKENS
from .context import Context


def dumpHttpFile(file: str, url: str, method: str, headers: Dict[str, str], obj: Any) -> None:
    try:
        json_str = json.dumps(obj, indent=2, ensure_ascii=False)
        with open(file, "w", encoding="utf-8") as f:
            f.write(f"{method.upper()} {url}\n")
            for key, value in headers.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write(json_str)
        print(f"HTTP request successfully dumped to {file}")
    except TypeError as e:
        print(f"Error: The object could not be serialized to JSON. Details: {e}")
    except OSError as e:
        print(f"Error: Could not write to file {file}. Details: {e}")


class ChatCompletionsClient:
    def __init__(self, api_key: str, model: str) -> None:
        if not (api_key and model):
            raise RuntimeError("OpenAI env missing. Set OPENAI_API_KEY and AI_MODEL.")
        self.base_url = "https://api.openai.com/v1"
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def call_responses(
        self,
        ctx: Context,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_schema: Dict[str, Any],
        max_completion_tokens: Optional[int] = None,
        interactive_tool_runner=None,
        message_sink=None,
        reasoning_effort: Optional[str] = "minimal",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Drop-in replacement that uses the Responses API instead of Chat Completions,
        while preserving the same signature and semantics expected by Orion.
        """
        # Build a stable, normalized sink
        def _sink(msg: Dict[str, Any]) -> None:
            if message_sink:
                message_sink(msg)

        model = model or self.model
        # Normalize request payload for the Responses API
        url = f"{self.base_url}/responses"  # /v1/responses
        max_output_tokens = max_completion_tokens or MAX_COMPLETION_TOKENS

        # Local, mutable message buffer
        local_messages = list(messages)
        have_tools = bool(tools)
        max_tool_turns = 12
        turns = 0

        def _make_payload() -> Dict[str, Any]:
            # Keep the same json_schema format you already use
            payload = {
                "model": model,
                "input": local_messages,  # same role/content items you were sending in `messages`
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "OrionSchema",
                        "schema": response_schema,
                        "strict": True,
                    }
                },
                "max_output_tokens": max_output_tokens,
                "reasoning": {"effort": reasoning_effort or "minimal"},
            }
            if have_tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            return payload

        def _extract_msg_obj(resp_obj: Dict[str, Any]) -> Dict[str, Any]:
            """
            Normalize a Responses API result into a {content: str|None, tool_calls: list} 'message'
            compatible with the rest of Orion's logic.
            """
            output = resp_obj.get("output")
            content_chunks: List[str] = []
            tool_calls = []

            if output and isinstance(output, list):
                for o in output:
                    if not isinstance(o, dict):
                        continue
                    _otype = o.get("type")
                    if _otype == "message":
                        _ct = o.get("content")
                        if isinstance(_ct, str):
                            content_chunks.append(_ct)
                        elif isinstance(_ct, list):
                            for item in _ct:
                                if isinstance(item, str):
                                    content_chunks.append(item)
                                elif isinstance(item, dict) and item.get("type", "") == "output_text":
                                    content_chunks.append(item.get("text", ""))
                    elif _otype == "tool_call":
                        tool_calls.append(o)

            content = "\n".join([c for c in content_chunks if isinstance(c, str)]) if content_chunks else None
            return {"content": content, "tool_calls": tool_calls}

        while True:
            ctx.log(f"Calling POST (tools={len(tools) if tools else 0})")
            r = self.session.post(url, json=_make_payload(), timeout=240)
            if r.status_code != 200:
                raise RuntimeError(f"Responses API error {r.status_code}: {r.text[:2000]}")

            resp = r.json()

            # Normalize the Responses payload into a chat-like message object
            msg_obj = _extract_msg_obj(resp)

            tool_calls = msg_obj.get("tool_calls") or []
            if tool_calls:
                # Record assistant turn with tool_calls (same as before)
                _sink({"role": "assistant", "content": msg_obj.get("content", None), "tool_calls": tool_calls})

                if interactive_tool_runner is None:
                    raise RuntimeError("Tool requested but no interactive_tool_runner provided.")

                turns += 1
                if turns > max_tool_turns:
                    raise RuntimeError("Exceeded max tool-call turns; aborting.")

                # Execute each tool call and append tool results to the conversation
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    args_text = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                    except Exception:
                        args = {}
                    # Run the tool
                    tool_output = interactive_tool_runner(name, args)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    }
                    local_messages.append(tool_msg)
                    _sink(tool_msg)

                # Also append the assistant turn (with tool_calls) to the history we’ll resend
                local_messages.append({
                    "role": "assistant",
                    "content": msg_obj.get("content", None),
                    "tool_calls": tool_calls,
                })
                # Loop back to let the model continue after tool outputs
                continue

            # No tool calls → this should be the final, strict JSON text per your schema
            final_text = msg_obj.get("content") or ""
            _sink({"role": "assistant", "content": final_text})

            try:
                final_json = json.loads(final_text)
            except Exception as e:
                # Provide some debugging context if the model didn't honor json_schema
                raise RuntimeError(
                    f"Failed to parse strict JSON from model output: {e}\nOutput:\n{final_text[:1000]}"
                )
            return final_json

    def call_chatcompletions(
        self,
        ctx: Context,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_schema: Dict[str, Any],
        max_completion_tokens: Optional[int] = None,
        interactive_tool_runner=None,
        message_sink=None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "OrionSchema",
                    "schema": response_schema,
                    "strict": True,
                },
            },
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload["max_completion_tokens"] = max_completion_tokens or MAX_COMPLETION_TOKENS

        url = self.chat_url()

        local_messages = list(messages)
        max_tool_turns = 12
        turns = 0

        def _sink(msg: Dict[str, Any]) -> None:
            if message_sink:
                message_sink(msg)

        while True:
            ctx.log(f"Calling POST (tools={len(tools) if tools else 0})")
            r = self.session.post(
                url,
                json={
                    "model": payload["model"],
                    "messages": local_messages,
                    "response_format": payload["response_format"],
                    **({"tools": payload["tools"], "tool_choice": payload.get("tool_choice")} if tools else {}),
                    "max_completion_tokens": payload["max_completion_tokens"],
                },
                timeout=480,
            )
            if r.status_code != 200:
                raise RuntimeError(f"Chat Completions API error {r.status_code}: {r.text[:2000]}")
            resp = r.json()
            choice = (resp.get("choices") or [{}])[0]
            msg_obj = choice.get("message", {}) or {}

            tool_calls = msg_obj.get("tool_calls") or []
            if tool_calls:
                local_messages.append({"role": "assistant", "content": msg_obj.get("content", None), "tool_calls": tool_calls})
                _sink({"role": "assistant", "content": msg_obj.get("content", None), "tool_calls": tool_calls})
                if interactive_tool_runner is None:
                    raise RuntimeError("Tool requested but no interactive_tool_runner provided.")
                turns += 1
                if turns > max_tool_turns:
                    raise RuntimeError("Exceeded max tool-call turns; aborting.")
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    args_text = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                    except Exception:
                        args = {}
                    tool_output = interactive_tool_runner(name, args)
                    local_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    })
                    _sink({"role": "tool", "tool_call_id": tc_id, "content": json.dumps(tool_output, ensure_ascii=False)})
                continue

            final_text = msg_obj.get("content") or ""
            _sink({"role": "assistant", "content": final_text})
            try:
                final_json = json.loads(final_text)
            except Exception as e:
                raise RuntimeError(f"Failed to parse strict JSON from model output: {e}\nOutput:\n{final_text[:1000]}")
            return final_json
