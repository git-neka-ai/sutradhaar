# orion: Replace clean_invalid_ref_nodes with _preprocess_for_openai; enforce required=all property keys and additionalProperties=false for objects; keep Responses-only client and improve strict schema compatibility.

import json
from typing import Any, Dict, List, Optional

import requests

from .config import MAX_COMPLETION_TOKENS
from .context import Context

from copy import deepcopy
from pydantic import BaseModel
# orion: Add pathlib and time for optional .httpcalls request/response logging to REST Client .http files.
import pathlib
import time
# orion: Import random to provide exponential backoff jitter for retries on timeouts and HTTP 5xx.
import random
# orion: Import os for environment-based provider detection and secrets.
import os

# orion: Introduce a centralized OpenAI schema preprocessor that (1) removes sibling keys alongside $ref and (2) enforces strict object-shape requirements by ensuring required includes all property keys and additionalProperties=False for any node with properties.

def _preprocess_for_openai(schema: dict) -> dict:
    """
    Prepare a JSON Schema for OpenAI strict validation:
    - If a node has "$ref", remove all sibling keys and keep only the $ref.
    - For any object node that defines "properties", ensure:
        * "required" exists and includes every property key (deterministic order)
        * "additionalProperties" is set to False
    The transformation is applied recursively across the schema tree.
    """
    cleaned = deepcopy(schema)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # If a $ref is present, drop all sibling keys for OpenAI compatibility
            if "$ref" in node:
                for k in list(node.keys()):
                    if k != "$ref":
                        node.pop(k, None)
                return  # Nothing else to do at this node

            # Enforce strict object shape where properties are specified
            props = node.get("properties")
            if isinstance(props, dict):
                # Build required = all property keys (union with existing, deduped, sorted for determinism)
                existing_req = node.get("required") or []
                try:
                    existing_set = set(existing_req)
                except TypeError:
                    existing_set = set()
                all_keys = set(props.keys()) | existing_set
                node["required"] = sorted(all_keys)
                node["additionalProperties"] = False

            # Recurse into all values
            for v in list(node.values()):
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(cleaned)
    return cleaned



class ChatCompletionsClient:
    # orion: Updated client to indicate Responses-only usage; Chat Completions helpers were removed during cleanup.
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a minimal HTTP client for OpenAI's Responses API with provider autodetection.

        Provider selection precedence (highest first):
          1) Constructor args (api_key/model/base_url)
          2) settings['api'] values (provider, api_key, model, base_url)
          3) Environment
             - OpenAI: OPENAI_API_KEY, AI_MODEL, OPENAI_BASE_URL
             - Azure:  AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL, AZURE_OPENAI_ENDPOINT

        Provider detection rules:
          - If settings['api']['provider'] is 'azure' or 'openai', use it.
          - Else if Azure env hints exist (AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT) or
            base_url looks like an Azure endpoint, use 'azure'.
          - Otherwise default to 'openai'.

        Base URL normalization:
          - OpenAI: default https://api.openai.com/v1 ("/v1" suffix ensured)
          - Azure:  {endpoint}/openai/v1 ("/openai/v1" suffix ensured)

        Auth headers:
          - OpenAI: Authorization: Bearer <OPENAI_API_KEY>
          - Azure:  api-key: <AZURE_OPENAI_API_KEY>
        """
        self.session = requests.Session()

        api_cfg = (settings or {}).get("api") if isinstance(settings, dict) else None
        api_cfg = api_cfg if isinstance(api_cfg, dict) else {}

        # Determine provider
        provider: Optional[str] = str(api_cfg.get("provider") or "").strip().lower() or None

        def _looks_like_azure(url: Optional[str]) -> bool:
            if not url:
                return False
            u = url.lower()
            return ("azure.com" in u) or ("/openai/" in u) or ("openai.azure.com" in u)

        if provider not in ("azure", "openai"):
            if (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_ENDPOINT")) or _looks_like_azure(base_url or api_cfg.get("base_url")):
                provider = "azure"
            else:
                provider = "openai"

        # Resolve values with precedence: args > settings > env
        resolved_model: Optional[str]
        resolved_api_key: Optional[str]
        resolved_base_url: Optional[str]

        if provider == "azure":
            resolved_api_key = api_key or api_cfg.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY")
            resolved_model = model or api_cfg.get("model") or os.environ.get("AZURE_OPENAI_MODEL")
            endpoint = base_url or api_cfg.get("base_url") or os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise RuntimeError("Azure provider selected but no endpoint provided (AZURE_OPENAI_ENDPOINT or settings.api.base_url or base_url arg).")
            # Normalize to {endpoint}/openai/v1
            endpoint = endpoint.rstrip("/")
            if not endpoint.endswith("/openai/v1"):
                if endpoint.endswith("/openai"):
                    endpoint = f"{endpoint}/v1"
                else:
                    endpoint = f"{endpoint}/openai/v1"
            resolved_base_url = endpoint
            if not resolved_api_key:
                raise RuntimeError("Azure provider selected but no API key provided (AZURE_OPENAI_API_KEY or settings.api.api_key or api_key arg).")
            if not resolved_model:
                raise RuntimeError("Azure provider selected but no model deployment provided (AZURE_OPENAI_MODEL or settings.api.model or model arg).")
            self.session.headers.update({
                "api-key": resolved_api_key,
                "Content-Type": "application/json",
            })
        else:  # openai
            resolved_api_key = api_key or api_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
            resolved_model = model or api_cfg.get("model") or os.environ.get("AI_MODEL") or "gpt-5"
            resolved_base_url = base_url or api_cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            # Ensure /v1 suffix
            resolved_base_url = resolved_base_url.rstrip("/")
            if not resolved_base_url.endswith("/v1"):
                resolved_base_url = f"{resolved_base_url}/v1"
            if not resolved_api_key:
                raise RuntimeError("OpenAI provider selected but no API key provided (OPENAI_API_KEY or settings.api.api_key or api_key arg).")
            self.session.headers.update({
                "Authorization": f"Bearer {resolved_api_key}",
                "Content-Type": "application/json",
            })

        # Finalize
        self.model = resolved_model  # OpenAI model id or Azure deployment name
        self.base_url = resolved_base_url
        self.provider = provider

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
        _timeout: int = 1800,
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
        # orion: Preprocess schema for OpenAI strict mode: clean $ref siblings and enforce required for all properties with additionalProperties=False.
        response_schema = _preprocess_for_openai(response_schema)

        # Build a stable, normalized sink so we don't branch on None repeatedly.
        def _sink(msg: Dict[str, Any]) -> None:
            if message_sink:
                message_sink(msg)

        model = model or self.model
        # orion: Responses endpoint uses /v1/responses with different payload keys than chat.completions.
        url = f"{self.base_url}/responses"  # /v1/responses or /openai/v1/responses
        max_output_tokens = max_completion_tokens or MAX_COMPLETION_TOKENS

        # orion: Work on a local copy of messages to append tool outputs and assistant echoes.
        local_messages = list(messages)
        max_tool_turns = 50
        turns = 0

        # orion: Helpers to locate/emit the latest system_state inside the in-flight message list.
        def _parse_system_state_from_msg(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                return None
            content = msg.get("content")
            if not isinstance(content, str):
                return None
            try:
                obj = json.loads(content)
            except Exception:
                return None
            if isinstance(obj, dict) and obj.get("type") == "system_state":
                return obj
            return None

        def _find_latest_system_state() -> Optional[Dict[str, Any]]:
            for m in reversed(local_messages):
                obj = _parse_system_state_from_msg(m)
                if obj is not None:
                    return obj
            return None

        # orion: Upsert system_state within the current call to keep exactly one system_state message in local_messages.
        # If a system_state exists, replace it in-place; otherwise append. We still sink to persist the upgrade across turns.
        def _append_system_state(obj: Dict[str, Any]) -> None:
            # orion: Build the canonical message and replace the existing system_state rather than appending duplicates.
            msg = {"type": "message", "role": "system", "content": json.dumps(obj, ensure_ascii=False)}
            # Find the most recent system_state message index (if any)
            idx: Optional[int] = None
            for i in range(len(local_messages) - 1, -1, -1):
                m = local_messages[i]
                if m.get("role") != "system":
                    continue
                content = m.get("content")
                if not isinstance(content, str):
                    continue
                try:
                    parsed = json.loads(content)
                except Exception:
                    continue
                if isinstance(parsed, dict) and parsed.get("type") == "system_state":
                    idx = i
                    break
            if idx is not None:
                # orion: Replace in-place to dedupe system_state for this in-flight call.
                local_messages[idx] = msg
            else:
                # orion: No prior system_state in this call; append the first one.
                local_messages.append(msg)
            # orion: Persist the upgraded system_state so subsequent turns start from the latest state.
            _sink(msg)

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
                payload["tools"].append({"type": "web_search"})
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

        # orion: Log token usage for every Responses API call; supports multiple schema variants for robustness.
        def _log_usage(resp_obj: Dict[str, Any]) -> None:
            usage = resp_obj.get("usage") or {}

            def _as_int(v: Any) -> int:
                try:
                    return int(v) if v is not None else 0
                except Exception:
                    return 0

            # Prefer Responses fields, fall back to Chat Completions aliases
            input_tokens = usage.get("input_tokens")
            if input_tokens is None:
                input_tokens = usage.get("prompt_tokens")

            output_tokens = usage.get("output_tokens")
            if output_tokens is None:
                output_tokens = usage.get("completion_tokens")

            # Cached input tokens may appear in several places
            cached_input = None
            itd = usage.get("input_token_details") or {}
            ptd = usage.get("prompt_tokens_details") or {}
            for cand in (
                itd.get("cached_tokens"),
                ptd.get("cached_tokens"),
                usage.get("cache_read_input_tokens"),
            ):
                if cand is not None:
                    cached_input = cand
                    break

            input_tokens_i = _as_int(input_tokens)
            output_tokens_i = _as_int(output_tokens)
            cached_input_i = _as_int(cached_input)

            try:
                ctx.log(
                    f"OpenAI usage: input_tokens={input_tokens_i}, cached_input_tokens={cached_input_i}, output_tokens={output_tokens_i}"
                )
            except Exception:
                # Best-effort logging; never fail the request loop
                pass

        while True:
            # orion: Build payload once per POST so logging captures the exact body sent.
            payload = _make_payload()

            # orion: If logging is enabled, write the request in REST Client format with redacted Authorization/api-key.
            try:
                http_file = None
                settings = getattr(ctx, "settings", {}) or {}
                log_cfg = {}
                if isinstance(settings, dict):
                    log_cfg = ((settings.get("logging") or {}).get("httpcalls") or {})
                enabled = log_cfg.get("enabled") if isinstance(log_cfg, dict) else None
                custom_dir = log_cfg.get("dir") if isinstance(log_cfg, dict) else None

                # Determine directory and behavior based on settings
                if enabled is False:
                    http_file = None  # explicit off
                else:
                    # Compute target directory
                    base_dir: Optional[pathlib.Path] = None
                    if enabled is True:
                        if custom_dir:
                            cpath = pathlib.Path(str(custom_dir))
                            base_dir = cpath if cpath.is_absolute() else (pathlib.Path(ctx.repo_root) / cpath)
                        else:
                            base_dir = pathlib.Path(ctx.repo_root) / ".httpcalls"
                        # Ensure directory exists when explicitly enabled
                        base_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        # Back-compat: only log if default dir already exists
                        maybe = pathlib.Path(ctx.repo_root) / ".httpcalls"
                        if maybe.exists() and maybe.is_dir():
                            base_dir = maybe

                    if base_dir is not None:
                        ts_ms = int(time.time() * 1000)
                        http_file_path = base_dir / f"call-{ts_ms}.http"
                        headers_for_log = dict(self.session.headers)
                        if "Authorization" in headers_for_log:
                            headers_for_log["Authorization"] = "Bearer {{OPENAI_API_KEY}}"
                        if "api-key" in headers_for_log:
                            headers_for_log["api-key"] = "{{AZURE_OPENAI_API_KEY}}"
                        dumpHttpFile(str(http_file_path), url, "POST", headers_for_log, payload)
                        http_file = http_file_path
            except Exception:
                http_file = None  # Non-fatal: proceed without logging

            # orion: Conservative timeout to accommodate tool loops; Responses may stream chunks server-side.
            # orion: Wrap the POST in a bounded retry loop for transient failures (timeouts, HTTP 5xx). 4xx errors are not retried.
            max_retries = 3
            attempt = 0
            last_exc: Optional[Exception] = None
            r = None
            while True:
                attempt += 1
                t0 = time.time()
                try:
                    r = self.session.post(url, json=payload, timeout=_timeout)
                    elapsed_ms = int((time.time() - t0) * 1000)

                    # orion: Append a response section to the same .http file with status, headers, body, and elapsed time (non-fatal on errors).
                    try:
                        if http_file is not None:
                            with open(http_file, "a", encoding="utf-8") as f:
                                f.write("\n\n### Response — elapsed_ms: " + str(elapsed_ms) + "\n")
                                # Best-effort HTTP status line; requests doesn't expose HTTP version reliably.
                                f.write(f"HTTP/1.1 {r.status_code} {getattr(r, 'reason', '')}\n")
                                for hk, hv in r.headers.items():
                                    try:
                                        f.write(f"{hk}: {hv}\n")
                                    except Exception:
                                        continue
                                f.write("\n")
                                try:
                                    f.write(r.text)
                                except Exception:
                                    # Fallback if body can't be decoded as text
                                    f.write("<binary/undecodable body>\n")
                    except Exception:
                        pass  # Non-fatal

                    # Handle status codes
                    if r.status_code == 200:
                        break  # success
                    if r.status_code >= 500 and attempt <= max_retries:
                        # orion: Retry on server errors with exponential backoff and jitter to smooth contention.
                        base_delay = [1.0, 2.0, 4.0][min(attempt - 1, 2)]
                        delay = base_delay * random.uniform(0.5, 1.5)
                        try:
                            ctx.log(f"Responses API attempt {attempt} received {r.status_code}; retrying in {delay:.2f}s...")
                        except Exception:
                            pass
                        time.sleep(delay)
                        continue
                    # Do not retry for 4xx; raise immediately with truncated body for diagnostics.
                    if r.status_code != 200:
                        raise RuntimeError(f"Responses API error {r.status_code}: {r.text[:2000]}")
                except requests.exceptions.Timeout as e:
                    last_exc = e
                    if attempt <= max_retries:
                        base_delay = [1.0, 2.0, 4.0][min(attempt - 1, 2)]
                        delay = base_delay * random.uniform(0.5, 1.5)
                        try:
                            ctx.log(f"Responses API timeout on attempt {attempt}; retrying in {delay:.2f}s...")
                        except Exception:
                            pass
                        time.sleep(delay)
                        continue
                    # Retries exhausted
                    raise RuntimeError(f"Responses API timeout after {attempt} attempt(s): {e}")
                # Exit inner retry loop on success
                break

            if r is None:
                raise RuntimeError("Responses API: no response object after retries.")

            resp = r.json()

            # orion: Log token usage for this call using robust parsing of usage fields.
            _log_usage(resp)

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

                    # orion: Always emit the function_call record.
                    _itc = { "type": "function_call", "name": name, "arguments": args_text, "call_id": tc_id }
                    _sink(_itc)
                    local_messages.append(_itc)

                    # orion: Intercept get_file_contents to mutate system_state instead of streaming large content.
                    output_to_emit: Any = tool_output
                    if name == "get_file_contents" and isinstance(tool_output, dict):
                        # Extract raw result regardless of envelope shape
                        result = tool_output.get("result") if "result" in tool_output else tool_output
                        path = None
                        content = None
                        line_count = None
                        if isinstance(result, dict):
                            path = result.get("path")
                            content = result.get("content")
                            line_count = result.get("line_count")
                        state_obj = _find_latest_system_state()
                        if path and isinstance(content, str) and isinstance(state_obj, dict):
                            files_map = state_obj.get("files") or {}
                            entry = files_map.get(path)
                            if isinstance(entry, dict) and entry.get("kind") == "full":
                                output_to_emit = {"status": "noop", "reason": "already_full", "path": path}
                            else:
                                # Promote to full in system_state
                                try:
                                    bcount = len(content.encode("utf-8"))
                                except Exception:
                                    bcount = 0
                                files_map[path] = {
                                    "kind": "full",
                                    "body": content,
                                    "meta": {"line_count": int(line_count) if line_count is not None else None, "bytes": bcount},
                                }
                                state_obj["files"] = files_map
                                try:
                                    state_obj["version"] = int(state_obj.get("version", 0) or 0) + 1
                                except Exception:
                                    state_obj["version"] = 1
                                _append_system_state(state_obj)
                                output_to_emit = {"status": "ok", "path": path, "note": "contents added to system_state"}
                        elif name == "get_file_contents":
                            # No system_state or malformed tool result: avoid streaming file content
                            output_to_emit = {"status": "noop", "reason": "no_system_state_or_malformed_result"}

                    _otc = { "type": "function_call_output", "call_id": tc_id, "output": json.dumps(output_to_emit, ensure_ascii=False) }
                    _sink(_otc)
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
