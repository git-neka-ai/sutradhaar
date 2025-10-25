# orion: Cleaned up the client to be Responses-only; removed Chat Completions helpers and updated docs to reduce maintenance and confusion.

import json
from typing import Any, Dict, List, Optional

import requests

from .config import MAX_COMPLETION_TOKENS
from .context import Context


class ChatCompletionsClient:
    # orion: Updated client to indicate Responses-only usage; Chat Completions helpers were removed during cleanup.
    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize a minimal HTTP client for OpenAI's Responses API.

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

    # orion: Expand docstring and comments; Responses API is the single entry point and supports iterative tool-call handling.
    def call_responses(
        self,
        ctx: Context,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_schema: Dict[str, Any],
        max_completion_tokens: Optional[int] = None,
        interactive_tool_runner=None,
        message_sink=None,
        call_type: str = "minimal",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the Responses API and normalize the result into a strict JSON object.

        The method enforces a strict json_schema for the final assistant content,
        supports function-calling via the tools argument, and iteratively handles
        tool-call turns until a final JSON payload is produced.

        Args:
            ctx: Logger/console context.
            messages: Conversation history in role/content form.
            tools: Optional tool definitions for function-calling.
            response_schema: JSON schema that the final response must satisfy.
            max_completion_tokens: Optional cap for the model's final output.
            interactive_tool_runner: Callable(name, args) used to execute tools.
            message_sink: Optional callback invoked with assistant/tool messages.
            call_type: OpenAI call type hint for the endpoint.
            model: Optional model override for this call.

        Returns:
            The strict JSON object produced by the model.

        Raises:
            RuntimeError: On non-200 HTTP responses, schema non-compliance, or
                exceeding the max tool-call loop turns.
        """
        reasoning_effort = "minimal" if call_type.endswith("_summary") else "medium"

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
        max_tool_turns = 50
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
            payload["tools"] = tools.copy() if tools else []
            payload["tool_choice"] = "auto"
            if reasoning_effort != "minimal":
                payload["tools"].append({"type":"web_search"})
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
                    elif _otype == "function_call":
                        tool_calls.append(o)

            content = "\n".join([c for c in content_chunks if isinstance(c, str)]) if content_chunks else None
            return {"content": content, "tool_calls": tool_calls}

        while True:
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

                if interactive_tool_runner is None:
                    raise RuntimeError("Tool requested but no interactive_tool_runner provided.")

                turns += 1
                if turns > max_tool_turns:
                    raise RuntimeError("Exceeded max tool-call turns; aborting.")

                # orion: Execute each tool call and append tool results to the conversation for the next model turn.
                for tc in tool_calls:
                    tc_id = tc.get("call_id")
                    name = tc.get("name")
                    args_text = tc.get("arguments", "{}")
                    try:
                        args = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                    except Exception:
                        # orion: Be forgiving with tool arg parsing; default to empty args if malformed.
                        args = {}
                    # Run the tool
                    ctx.log(f"Invoking tool: {name} with args: {json.dumps(args)}")
                    tool_output = interactive_tool_runner(name, args)
                    _itc = { "type": "function_call", "name": name, "arguments": args_text, "call_id": tc_id }
                    _otc = { "type": "function_call_output", "call_id": tc_id, "output": json.dumps(tool_output, ensure_ascii=False) }
                    _sink(_itc)
                    _sink(_otc)
                    local_messages.append(_itc)
                    local_messages.append(_otc)

                # orion: Loop back to let the model continue after tool outputs are present in context.
                continue

            # orion: No tool calls → expect the final, strict JSON text per the provided schema.
            final_text = msg_obj.get("content") or ""
            _sink({"type": "message", "role": "assistant", "content": final_text})

            try:
                final_json = json.loads(final_text)
            except Exception as e:
                # orion: Provide context if the model omitted or corrupted strict JSON.
                raise RuntimeError(
                    f"Failed to parse strict JSON from model output: {e}\nOutput:\n{final_text[:1000]}"
                )
            return final_json



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

