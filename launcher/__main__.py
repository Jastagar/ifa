import argparse
import os
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from ifa.services.pairing import lan_address
from launcher.server import Launcher, serve
from launcher.settings_schema import BY_KEY
from launcher import env_file
from launcher.system_info import check_ollama, venv_python

REPO_ROOT = Path(__file__).resolve().parent.parent
LAN_KEY_PATH = REPO_ROOT / ".launcher_key"


def lan_key() -> str:
    """Keep one access key across restarts, so a bookmarked LAN link keeps working."""
    key = LAN_KEY_PATH.read_text(encoding="utf-8").strip() if LAN_KEY_PATH.exists() else ""
    if not key:
        key = secrets.token_urlsafe(12)
        LAN_KEY_PATH.write_text(key, encoding="utf-8")
    return key


def share_ollama(ip: str) -> None:
    if check_ollama(f"http://{ip}:11434", "")["ok"]:
        print(f"Ollama is open on the network at http://{ip}:11434")
    elif check_ollama("http://127.0.0.1:11434", "")["ok"]:
        print("Ollama is running for this PC only. To open it on the network: quit Ollama "
              "(tray icon -> Quit), then start this launcher again with --lan.")
    else:
        kwargs: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS} \
            if sys.platform == "win32" else {"start_new_session": True}
        try:
            subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"},
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
            print(f"Started Ollama on the network at http://{ip}:11434")
        except OSError as exc:
            print(f"Couldn't start Ollama: {exc}")


def auto_launch(launcher: Launcher) -> None:
    """Start Ifa right away, like pressing Launch, when everything it needs is in place."""
    if venv_python(REPO_ROOT) is None:
        print("Not starting Ifa yet: run the Setup steps in the launcher first.")
        return
    values = env_file.read_env(launcher.env_path)
    if not values.get("IFA_OLLAMA_MODEL", "").strip():
        print("Not starting Ifa yet: set the model (Language model -> Model) and press Launch.")
        return
    # Ifa quits at startup if Ollama doesn't answer, and Ollama may have only just been started.
    url = values.get("MAIN_OLLAMA_URL") or BY_KEY["MAIN_OLLAMA_URL"].default
    for _ in range(20):
        if check_ollama(url, "")["ok"]:
            break
        time.sleep(1)
    else:
        print(f"Not starting Ifa: Ollama isn't answering at {url}. Start Ollama, then press Launch.")
        return
    try:
        launcher.start_agent()
        print("Starting Ifa (use --no-launch to only open the launcher).")
    except Exception as exc:
        print(f"Couldn't start Ifa automatically: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m launcher", description="Ifa launcher")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-launch", action="store_true", help="open the launcher without starting Ifa")
    parser.add_argument("--lan", action="store_true",
                        help="also open the launcher (with an access key) and Ollama to other devices on the network")
    args = parser.parse_args()

    launcher = Launcher(REPO_ROOT)
    ip = lan_address() if args.lan else None
    if args.lan and not ip:
        print("No network address found; running for this PC only.")
    key = lan_key() if ip else None
    server, token = serve(launcher, args.port, lan_key=key, lan_host=ip)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Ifa launcher running at {url}  (Ctrl-C to quit)")
    if ip:
        print(f"On the network: http://{ip}:{args.port}/?key={key}")
        print("Windows may ask to allow Python (and Ollama) through the firewall: allow it on private networks.")
        share_ollama(ip)
    if not args.no_launch:
        threading.Thread(target=auto_launch, args=(launcher,), daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        launcher.agent.stop()
        server.server_close()


if __name__ == "__main__":
    main()
