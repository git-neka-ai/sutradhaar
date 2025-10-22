# orion: Isolated the OpenAI client wrapper (Responses and Chat Completions compatibility) and HTTP dump helper to keep API concerns separate from Orion logic. Expanded docstrings and inline comments to clarify payload construction, tool-call loop behavior, and timeout/error handling.

import json
from typing import Any, Dict, List, Optional

import requests

from .config import MAX_COMPLETION_TOKENS
from .context import Context


# orion: Add a docstring to clarify intent, parameters, and failure behavior (non-throwing; logs to stderr via prints).
def dumpHttpFile(file: str, url: str, method: str, headers: Dict[str, str], obj: Any) -> None:
    """
    Write a human-readable HTTP request dump to disk for debugging.

    Args:
        file: Destination file path for the dump.
        url: The target URL of the request.
        method: HTTP verb (GET/POST/...).
        headers: Request headers that will be sent.
        obj: JSON-serializable body object that will be pretty-printed.

    Notes:
        - This helper is best-effort: it catches serialization and I/O errors and
          prints a descriptive message instead of raising.
        - The object is serialized with ensure_ascii=False to preserve unicode.
    """
    try:
        # orion: Serialize payload deterministically with indentation for readability.
        json_str = json.dumps(obj, indent=2, ensure_ascii=False)
        with open(file, "w", encoding="utf-8") as f:
            # orion: Write request line followed by headers and body for standard debugging format.
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
    # orion: Add a docstring to the client to explain supported endpoints and configuration.
    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize a minimal HTTP client for OpenAI's Chat Completions and Responses APIs.

        Args:
            api_key: Secret API key for authentication.
            model: Default model name to use when a call does not override it.

        Raises:
            RuntimeError: If api_key or model are not provided.
        """
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

    # orion: Document helper returning the canonical chat completions endpoint.
    def chat_url(self) -> str:
        """Return the base Chat Completions endpoint URL."""
        return f"{self.base_url}/chat/completions"

    # orion: Expand docstring and comments to clarify how Responses API is normalized and how tool calls are handled iteratively.
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
        Invoke the Responses API and normalize the result into a strict JSON object.

        This is a drop-in replacement mirroring call_chatcompletions semantics,
        including iterative tool-call handling. It enforces a strict json_schema
        for the final assistant content and returns the parsed object.

        Args:
            ctx: Logger/console context.
            messages: Conversation history in role/content form.
            tools: Optional tool definitions for function-calling.
            response_schema: JSON schema that the final response must satisfy.
            max_completion_tokens: Optional cap for the model's final output.
            interactive_tool_runner: Callable(name, args) used to execute tools.
            message_sink: Optional callback invoked with assistant/tool messages.
            reasoning_effort: OpenAI reasoning effort hint for the endpoint.
            model: Optional model override for this call.

        Returns:
            The strict JSON object produced by the model.

        Raises:
            RuntimeError: On non-200 HTTP responses, schema non-compliance, or
                exceeding the max tool-call loop turns.
        """
        # Build a stable, normalized sink so we don't branch on None repeatedly.
        def _sink(msg: Dict[str, Any]) -> None:
            if message_sink:
                message_sink(msg)

        model = model or self.model
        # orion: Responses endpoint uses /v1/responses with different payload keys than chat.completions.
        url = f"{self.base_url}/responses"  # /v1/responses
        max_output_tokens = max_completion_tokens or MAX_COMPLETION_TOKENS

        # orion: Work on a local copy of messages to append tool outputs and assistant echoes.
        local_messages = list(messages)
        have_tools = bool(tools)
        max_tool_turns = 12
        turns = 0

        def _make_payload() -> Dict[str, Any]:
            # orion: Keep the same json_schema format you already use; Responses nests it under text.format.
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
            Normalize a Responses API result into a chat-like message object.

            The returned object contains only two fields that matter to Orion's
            control flow: a content string (final JSON text) and tool_calls.
            """
            output = resp_obj.get("output")
            content_chunks: List[str] = []
            tool_calls = []

            # orion: Responses API streams a heterogeneous list; we stitch content while collecting tool_call objects.
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
            # orion: Conservative timeout to accommodate tool loops; Responses may stream chunks server-side.
            r = self.session.post(url, json=_make_payload(), timeout=240)
            if r.status_code != 200:
                # orion: Surface first 2KB of body for fast diagnostics without overwhelming logs.
                raise RuntimeError(f"Responses API error {r.status_code}: {r.text[:2000]}")

            resp = r.json()

            # orion: Normalize the Responses payload into a chat-like message object for unified downstream handling.
            msg_obj = _extract_msg_obj(resp)

            tool_calls = msg_obj.get("tool_calls") or []
            if tool_calls:
                # orion: Record assistant turn with tool_calls for correct replay context in subsequent turns.
                _sink({"role": "assistant", "content": msg_obj.get("content", None), "tool_calls": tool_calls})

                if interactive_tool_runner is None:
                    raise RuntimeError("Tool requested but no interactive_tool_runner provided.")

                turns += 1
                if turns > max_tool_turns:
                    raise RuntimeError("Exceeded max tool-call turns; aborting.")

                # orion: Execute each tool call and append tool results to the conversation for the next model turn.
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    args_text = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                    except Exception:
                        # orion: Be forgiving with tool arg parsing; default to empty args if malformed.
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

                # orion: Also append the assistant turn (with tool_calls) to the history we’ll resend.
                local_messages.append({
                    "role": "assistant",
                    "content": msg_obj.get("content", None),
                    "tool_calls": tool_calls,
                })
                # orion: Loop back to let the model continue after tool outputs are present in context.
                continue

            # orion: No tool calls → expect the final, strict JSON text per the provided schema.
            final_text = msg_obj.get("content") or ""
            _sink({"role": "assistant", "content": final_text})

            try:
                final_json = json.loads(final_text)
            except Exception as e:
                # orion: Provide context if the model omitted or corrupted strict JSON.
                raise RuntimeError(
                    f"Failed to parse strict JSON from model output: {e}\nOutput:\n{final_text[:1000]}"
                )
            return final_json

    # orion: Add docstring and comments to clarify payload fields, tool loop, and timeout semantics.
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
        """
        Invoke the Chat Completions API and return the strict JSON object produced by the model.

        Mirrors the behavior of call_responses by iterating tool calls until the
        assistant returns a final JSON string adhering to the provided schema.
        """
        # orion: Build payload with response_format.json_schema for strict JSON, and optional tools for auto function-calling.
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

        # orion: Maintain our own local history to capture tool outputs and assistant tool calls between turns.
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
                # orion: Slightly higher timeout here to account for stricter schema validation and tool loops.
                timeout=480,
            )
            if r.status_code != 200:
                raise RuntimeError(f"Chat Completions API error {r.status_code}: {r.text[:2000]}")
            resp = r.json()
            choice = (resp.get("choices") or [{}])[0]
            msg_obj = choice.get("message", {}) or {}

            tool_calls = msg_obj.get("tool_calls") or []
            if tool_calls:
                # orion: Echo assistant tool_calls and append for context in the next turn.
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
                # orion: Continue loop to let the model incorporate tool outputs.
                continue

            # orion: Final assistant message is expected to be strict JSON text.
            final_text = msg_obj.get("content") or ""
            _sink({"role": "assistant", "content": final_text})
            try:
                final_json = json.loads(final_text)
            except Exception as e:
                raise RuntimeError(f"Failed to parse strict JSON from model output: {e}\nOutput:\n{final_text[:1000]}")
            return final_json
