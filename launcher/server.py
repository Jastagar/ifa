from __future__ import annotations

import hmac
import json
import re
import secrets
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ifa.services import custom_apps, memory_store, prompts
from ifa.skills.enums import App
from launcher import env_file
from launcher.agent_process import AgentProcess, LogBuffer
from launcher.setup_steps import SetupRunner, display, plan
from launcher.settings_schema import BY_KEY, GROUPS, SETTINGS, normalize
from launcher.system_info import check_ollama, detect_system, venv_python
from ifa.services.pairing import lan_address, load_pairing_code

STATIC = Path(__file__).parent / "static"
CUSTOM_KEY = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
MAX_BODY = 64_000


class Launcher:
    def __init__(self, repo_root: Path, agent: AgentProcess | None = None) -> None:
        self.repo_root = repo_root
        self.env_path = repo_root / ".env"
        self.apps_path = repo_root / "apps.json"
        self.prompts_path = repo_root / "prompts.json"
        self.db_path = str(repo_root / "ifa.db")
        self.log = agent.log if agent else LogBuffer(log_dir=repo_root / "logs")
        python = venv_python(repo_root) or Path(sys.executable)
        self.agent = agent or AgentProcess([str(python), "-u", "-m", "ifa.main"], cwd=str(repo_root), log=self.log)
        self.setup = SetupRunner(repo_root, AgentProcess([], cwd=str(repo_root), log=self.log, name="setup"), self._plan)
        self._env_lock = threading.Lock()

    def _plan(self):
        return plan(self.repo_root, env_file.read_env(self.env_path), BY_KEY["MAIN_OLLAMA_URL"].default)

    def setup_steps(self) -> dict:
        steps = [
            {"id": s.id, "label": s.label, "help": s.help, "command": display(s.argv, self.repo_root) if s.argv else "",
             "done": s.done, "blocked": s.blocked, "rerun": s.rerun}
            for s in self._plan()
        ]
        return {"steps": steps, **self.setup.status()}

    def run_setup(self, body: dict) -> dict:
        if self.agent.status()["state"] == "running":
            raise RuntimeError("stop the agent before running setup")
        ids = body.get("steps")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError("steps must be a list of step ids")
        self.setup.run(ids, only_missing=bool(body.get("only_missing")))
        return self.setup.status()

    def status(self) -> dict:
        return {**self.agent.status(), "setup": self.setup.status()}

    def settings(self) -> dict:
        values = env_file.read_env(self.env_path)
        groups = []
        for group_id, label in GROUPS:
            fields = []
            for s in (s for s in SETTINGS if s.group == group_id):
                field = {
                    "key": s.key, "label": s.label, "kind": s.kind, "default": s.default,
                    "help": s.help, "options": list(s.options), "minimum": s.minimum, "maximum": s.maximum,
                }
                if s.secret:
                    field["is_set"] = bool(values.get(s.key))
                else:
                    field["value"] = values.get(s.key, "")
                fields.append(field)
            groups.append({"id": group_id, "label": label, "settings": fields})
        custom = sorted(k for k in values if k not in BY_KEY)
        return {"groups": groups, "custom_keys": custom, "env_path": str(self.env_path)}

    def save(self, payload: dict) -> dict:
        values = payload.get("values") or {}
        custom = payload.get("custom") or {}
        if not isinstance(values, dict) or not isinstance(custom, dict):
            raise ValueError("values and custom must be objects")

        changes: dict[str, str | None] = {}
        errors: dict[str, str] = {}
        for key, raw in values.items():
            setting = BY_KEY.get(key)
            if setting is None:
                errors[key] = "unknown setting"
                continue
            if raw is None:
                continue
            if not isinstance(raw, str):
                errors[key] = "must be text"
                continue
            try:
                value = normalize(setting, raw)
            except ValueError as exc:
                errors[key] = str(exc)
                continue
            changes[key] = value or None

        for key, raw in custom.items():
            if key in BY_KEY or not CUSTOM_KEY.match(key):
                errors[key] = "use UPPER_CASE letters, digits and _ (not a built-in setting)"
            elif raw is not None and (not isinstance(raw, str) or "\n" in raw or "\r" in raw):
                errors[key] = "must be a single line of text"
            else:
                changes[key] = raw if raw else None

        if errors:
            raise SettingsError(errors)
        with self._env_lock:
            env_file.update_env(self.env_path, changes)
        return {**self.settings(), "restart_needed": self.agent.status()["state"] == "running"}

    def apps(self) -> dict:
        return {"apps": custom_apps.load(self.apps_path), "builtin": [a.value for a in App]}

    def prompts(self) -> dict:
        overrides = prompts.load(self.prompts_path)
        return {"prompts": [
            {"key": key, "label": label, "help": help, "default": default, "value": overrides.get(key, default)}
            for key, label, help, default in prompts.PROMPTS
        ]}

    def save_prompts(self, body: dict) -> dict:
        values, errors = prompts.validate(body.get("prompts"))
        if errors:
            raise SettingsError(errors)
        prompts.save(values, self.prompts_path)
        return self.prompts()

    def memories(self) -> dict:
        rows = memory_store.all_memories(self.db_path)
        return {"kinds": list(memory_store.KINDS), "memories": [
            {"id": r["id"], "topic": r["topic"], "fact": r["fact"], "kind": r["kind"],
             "updated": memory_store.ago(r["updated_at"])}
            for r in rows
        ]}

    def save_memories(self, body: dict) -> dict:
        items, errors = memory_store.validate(body.get("memories"))
        if errors:
            raise SettingsError({str(i): message for i, message in errors.items()})
        memory_store.replace_all(self.db_path, items)
        return self.memories()

    def save_apps(self, body: dict) -> dict:
        apps, errors = custom_apps.validate(body.get("apps"))
        if errors:
            raise SettingsError({str(i): message for i, message in errors.items()})
        custom_apps.save(apps, self.apps_path)
        return {**self.apps(), "restart_needed": self.agent.status()["state"] == "running"}

    def checks(self) -> dict:
        values = env_file.read_env(self.env_path)
        model_url = values.get("MAIN_OLLAMA_URL") or BY_KEY["MAIN_OLLAMA_URL"].default
        return {
            "system": detect_system(self.repo_root),
            "ollama": check_ollama(model_url, values.get("IFA_OLLAMA_MODEL", "")),
            "env_file": self.env_path.exists(),
            "phone": self.phone_info(values),
        }

    def share_logs(self) -> dict:
        from launcher.log_share import build_report, upload

        values = env_file.read_env(self.env_path)
        secret_keys = [s.key for s in SETTINGS if s.secret] + [k for k in values if k not in BY_KEY]
        report = build_report(self.log.text(), self.checks(), self.setup_steps(),
                              [values[k] for k in secret_keys if values.get(k)])
        url = upload(report)
        return {"url": url, "id": url.rstrip("/").rsplit("/", 1)[-1], "lines": report.count("\n")}

    def phone_info(self, values: dict) -> dict:
        enabled = (values.get("IFA_PHONE_ENABLED") or "1").lower() not in {"0", "false", "no", "off"}
        if not enabled:
            return {"enabled": False}
        try:
            code = load_pairing_code(self.repo_root / ".phone_token", values.get("IFA_PHONE_TOKEN", ""))
        except OSError as exc:
            return {"enabled": True, "error": str(exc)}
        port = values.get("IFA_PHONE_PORT") or BY_KEY["IFA_PHONE_PORT"].default
        return {"enabled": True, "pairing_code": code, "address": lan_address(), "port": port}

    def start_agent(self) -> dict:
        if self.setup.status()["running"]:
            raise RuntimeError("wait for setup to finish before launching")
        managed = set(BY_KEY) | set(env_file.read_env(self.env_path))
        self.agent.start(clear_env=managed)
        return self.agent.status()


class SettingsError(ValueError):
    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("invalid settings")
        self.errors = errors


def make_handler(launcher: Launcher, token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    page = (STATIC / "index.html").read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            pass

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, data: dict) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

        def _guard(self) -> bool:
            if self.headers.get("Host") not in allowed_hosts:
                self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
                return False
            path = urlparse(self.path).path
            if path.startswith("/api/") and not hmac.compare_digest(
                self.headers.get("X-Launcher-Token", ""), token
            ):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong launcher token"})
                return False
            return True

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            return data

        def do_GET(self) -> None:
            if not self._guard():
                return
            url = urlparse(self.path)
            if url.path == "/":
                html = page.replace("__LAUNCHER_TOKEN__", token)
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/settings":
                self._json(HTTPStatus.OK, launcher.settings())
            elif url.path == "/api/checks":
                self._json(HTTPStatus.OK, launcher.checks())
            elif url.path == "/api/status":
                self._json(HTTPStatus.OK, launcher.status())
            elif url.path == "/api/apps":
                self._json(HTTPStatus.OK, launcher.apps())
            elif url.path == "/api/prompts":
                self._json(HTTPStatus.OK, launcher.prompts())
            elif url.path == "/api/memory":
                self._json(HTTPStatus.OK, launcher.memories())
            elif url.path == "/api/setup":
                self._json(HTTPStatus.OK, launcher.setup_steps())
            elif url.path == "/api/logs":
                after = int((parse_qs(url.query).get("after") or ["0"])[0])
                lines, last = launcher.log.after(after)
                self._json(HTTPStatus.OK, {"lines": lines, "last": last, **launcher.status()})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._guard():
                return
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/settings":
                    self._json(HTTPStatus.OK, launcher.save(body))
                elif path == "/api/agent/start":
                    self._json(HTTPStatus.OK, launcher.start_agent())
                elif path == "/api/agent/stop":
                    launcher.agent.stop()
                    self._json(HTTPStatus.OK, launcher.agent.status())
                elif path == "/api/apps":
                    self._json(HTTPStatus.OK, launcher.save_apps(body))
                elif path == "/api/prompts":
                    self._json(HTTPStatus.OK, launcher.save_prompts(body))
                elif path == "/api/memory":
                    self._json(HTTPStatus.OK, launcher.save_memories(body))
                elif path == "/api/setup/run":
                    self._json(HTTPStatus.OK, launcher.run_setup(body))
                elif path == "/api/logs/share":
                    try:
                        self._json(HTTPStatus.OK, launcher.share_logs())
                    except OSError as exc:
                        self._json(HTTPStatus.BAD_GATEWAY, {"error": f"upload failed: {exc}"})
                elif path == "/api/setup/stop":
                    launcher.setup.stop()
                    self._json(HTTPStatus.OK, launcher.setup.status())
                elif path == "/api/agent/input":
                    launcher.agent.send(str(body.get("text", "")))
                    self._json(HTTPStatus.OK, {"ok": True})
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except SettingsError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc), "fields": exc.errors})
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def serve(launcher: Launcher, port: int, token: str | None = None) -> tuple[ThreadingHTTPServer, str]:
    token = token or secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    server.RequestHandlerClass = make_handler(launcher, token, server.server_address[1])
    return server, token
