"""`call_n8n_workflow` tool — outbound webhook POST to user-configured n8n workflows.

Configuration lives in `ifa/config/n8n_workflows.yaml` (gitignored; a
template `.example` file is committed). Each workflow has a URL and
optional auth (with env-var credential refs), timeout, and payload JSON
Schema.

Security posture (R17/R18):
- Response body truncated to 2KB before the agent loop delimits it
- LLM-supplied payload is validated against the workflow's `payload_schema`;
  default is an empty-properties object with `additionalProperties: false`
  (LLM can only send {} unless the workflow explicitly declares fields)
- Auth secret resolved at call time from environment; never stored in the
  agent message list or memory; system prompt instructs the LLM not to
  echo authentication values from tool results
"""
import os
from pathlib import Path

import httpx
import yaml
from jsonschema import ValidationError, validate

from ifa.core.context import AgentContext
from ifa.tools.registry import Tool, register

DEFAULT_TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 2048
DEFAULT_PAYLOAD_SCHEMA: dict = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class N8nConfigError(Exception):
    """Raised when the n8n YAML config can't be parsed or has invalid shape."""


def load_n8n_config(path: str | Path) -> dict:
    """Load and validate the n8n workflows YAML.

    Returns a dict `{workflow_name: {url, auth?, timeout?, payload_schema?}}`.
    Missing file returns an empty dict (Ifa boots with no workflows — any
    `call_n8n_workflow` call errors gracefully). YAML parse errors or
    structural problems raise N8nConfigError with an actionable message.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise N8nConfigError(
            f"n8n config at `{path}` has a YAML syntax error: {exc}. "
            "Fix the file or delete it to boot with no workflows."
        ) from exc

    if not isinstance(raw, dict) or "workflows" not in raw:
        raise N8nConfigError(
            f"n8n config at `{path}` must have a top-level `workflows:` key."
        )

    workflows = raw["workflows"] or {}
    if not isinstance(workflows, dict):
        raise N8nConfigError("`workflows` must be a mapping of name → config.")

    # Validate each workflow
    normalized = {}
    for name, config in workflows.items():
        if not isinstance(config, dict):
            raise N8nConfigError(f"workflow `{name}` must be a mapping.")
        if "url" not in config:
            raise N8nConfigError(f"workflow `{name}` is missing required `url` field.")
        normalized[name] = config

    return normalized


def _resolve_auth_header(auth_config: dict) -> tuple[str, str] | None:
    """Return (header_name, header_value) or None if no auth is configured.

    Raises ValueError if the auth block is malformed or the env var is unset.
    """
    auth_type = auth_config.get("type")
    if auth_type == "header":
        header_name = auth_config.get("name")
        env_var = auth_config.get("env")
        if not header_name or not env_var:
            raise ValueError("header auth requires `name` and `env` fields")
        value = os.environ.get(env_var)
        if value is None:
            raise ValueError(f"env var `{env_var}` is not set")
        return header_name, value
    if auth_type == "basic":
        # Not supported in Stage 1 — keep the schema extensible but reject clearly
        raise ValueError("basic auth not supported in Stage 1; use type=header")
    raise ValueError(f"unknown auth type `{auth_type}`")


def _handler(args: dict, ctx: AgentContext) -> str:
    workflow_name = args["workflow_name"]
    payload = args.get("payload", {})

    workflow = ctx.n8n_config.get(workflow_name)
    if workflow is None:
        return (
            f"No workflow named `{workflow_name}` is configured. "
            f"Available: {sorted(ctx.n8n_config.keys()) or 'none'}"
        )

    # Validate payload against the workflow's schema (defense: R18)
    schema = workflow.get("payload_schema", DEFAULT_PAYLOAD_SCHEMA)
    try:
        validate(instance=payload, schema=schema)
    except ValidationError as exc:
        return f"Payload rejected by schema for workflow `{workflow_name}`: {exc.message}"

    # Resolve auth header (if configured)
    headers = {}
    auth_config = workflow.get("auth")
    if auth_config is not None:
        try:
            header_name, header_value = _resolve_auth_header(auth_config)
            headers[header_name] = header_value
        except ValueError as exc:
            return f"Auth misconfigured for workflow `{workflow_name}`: {exc}"

    timeout = workflow.get("timeout", DEFAULT_TIMEOUT_SECONDS)

    try:
        response = httpx.post(
            workflow["url"],
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except httpx.TimeoutException:
        return f"Workflow `{workflow_name}` timed out after {timeout}s."
    except httpx.ConnectError as exc:
        return f"Workflow `{workflow_name}` is unreachable: {exc}"
    except httpx.HTTPError as exc:
        return f"Workflow `{workflow_name}` failed: {exc}"

    # Truncate before returning — prevents context-window pollution
    body = response.text
    if len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        body = body.encode("utf-8")[:MAX_RESPONSE_BYTES].decode("utf-8", errors="ignore")
        body = body + "\n... [response truncated at 2KB]"

    return f"Status {response.status_code}. Body:\n{body}"


TOOL = Tool(
    name="call_n8n_workflow",
    description=(
        "Trigger a named n8n workflow by POSTing a JSON payload to its webhook. "
        "Use when the user asks to run an automation, send a notification, or "
        "perform an action defined in their n8n instance."
    ),
    parameters={
        "type": "object",
        "properties": {
            "workflow_name": {
                "type": "string",
                "description": "The name of the configured workflow to invoke.",
            },
            "payload": {
                "type": "object",
                "description": "JSON object sent as the request body. Shape depends on the workflow.",
            },
        },
        "required": ["workflow_name", "payload"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)
