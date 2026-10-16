import argparse
import threading
import webbrowser
from pathlib import Path

from launcher.server import Launcher, serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m launcher", description="Ifa launcher")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    launcher = Launcher(Path(__file__).resolve().parent.parent)
    server, token = serve(launcher, args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Ifa launcher running at {url}  (Ctrl-C to quit)")
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
