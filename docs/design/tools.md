# Tools added or reworked

The tools whose design needed real decisions: Spotify, apps, websites, terminal commands and WhatsApp. For how tools plug into the loop, see [agent-loop.md](agent-loop.md).

---

## Spotify (`spotify_control`)

**Files:** [ifa/skills/spotify.py](../../ifa/skills/spotify.py), [ifa/tools/spotify.py](../../ifa/tools/spotify.py)

- **Login is a separate one-time step** (launcher Setup → Log in to Spotify, or `python -m ifa.skills.spotify`). The token is saved to `.spotify_token` at the repo root.
  - **Why:** the token used to live in `.cache` in whatever folder Ifa ran from. When we stopped tracking `.cache` in git, pulling deleted it on the PC. Every call then tried to open a browser login in the middle of a voice turn and failed. A tool call now never starts a login; it says how to log in.
- **Redirect URI:** `http://127.0.0.1:8888/callback`, which must also be listed in the Spotify dashboard. Spotify rejects `localhost` since 2025, but accepts `127.0.0.1`.
- **Errors become sentences:** wrong keys, expired login, Premium required, no device, rate limited, offline.
- **Two steps: search, then play** (you asked for this, because Hindi and Punjabi picks were bad).
  1. `search` returns up to 6 numbered results, ranked in code by:
     - title similarity and artist match
     - popularity and Spotify's own order
     - remix, slowed/reverb, lofi and cover uploads pushed down, unless you asked for one
  2. The model plays one by number: `play` with `choice`.
  - Searches use your account's country (`market=from_token`), and a named artist becomes a `track:… artist:…` filter with a plain fallback.
  - **Why:** the first result for Hindi songs was often a "(Slowed + Reverb)" upload.
  - **Cost:** one extra model step, so "open Spotify and play X" is exactly 3 tool steps (the limit).
- **Waits up to 8 s for a device**, so "open Spotify and play X" works while the app is still starting.

## Apps (`open_app`)

**Files:** [ifa/tools/system.py](../../ifa/tools/system.py), [ifa/skills/system.py](../../ifa/skills/system.py), [ifa/services/custom_apps.py](../../ifa/services/custom_apps.py)

- **Your own apps:** the launcher's "Apps Ifa can open" card stores a name plus a command, path or link (e.g. `steam://rungameid/570`) in `apps.json` (gitignored). An app with a built-in name replaces the built-in one.
- **The model picks a name from an enum; the command comes from your file.** The model never writes the command, which is the safe allowlist pattern.
- **Why not a second tool?** Two tools that both open apps would make the model guess between them.
- **The description says** to open without asking, and that "open X and do Y" means calling `open_app` first. That took "open Spotify and play X" from 15/16 to 20/20.

## Websites and YouTube (`open_website`, other session `4e5ac35`)

YouTube (plays the top real video, skipping ads), Google search or any URL, optionally private. **The URL is built in code, not by the model.** On Windows, Brave now gets the URL (`os.startfile` dropped it).

## Terminal commands (`run_command`)

**Files:** [ifa/skills/commands.py](../../ifa/skills/commands.py), [ifa/tools/commands.py](../../ifa/tools/commands.py)

Runs one command in `cmd.exe` (Windows) or the shell (macOS), and reads the output back.

- **Confirmation is enforced in code, not the prompt** (you chose "any command, ask first"). A command runs only when all of these hold:
  1. It was proposed in the previous turn. The first call always returns `NOT RUN YET` and Ifa asks.
  2. Your **very next** message is a yes ("yes", "okay", "go ahead", "haan", "theek hai"…). Anything with "no", "wait" or "nahi" doesn't count.
  3. The model asks again for the **same** command within 2 minutes.
- **Why code, not prompt:** prompt rules are suggestions, and the model reads third-party text such as notifications. A notification can talk the model into proposing `del /s`, but it can't make your next message say yes. Speaker verification then makes sure that yes is your voice.
- **Refused outright,** even after a yes: recursive deletes, formatting disks, shutdown and restart, registry deletes, account and permission changes, and piping downloaded code into a shell. The deny list is a backstop; confirmation is the real control.
- **Limits:** output is cut to the last 1,500 characters, and commands are killed (the whole process tree) after `IFA_COMMAND_TIMEOUT_SEC` (30). `IFA_COMMANDS_ENABLED=0` removes the tool.
- **Prompt:** prefer simple commands, and describe them in plain words when asking. One early run proposed a 120-character `awk` pipeline and read it aloud.

## WhatsApp (`whats_app`)

`pywhatkit` is imported only when sending. **Why:** importing it pinged google.com and raised an error offline, which stopped Ifa from starting.

## Next

- Prefer this computer as the Spotify device right after `open_app` (otherwise an active phone keeps playing).
- Saved, named commands that run without asking ("flush dns"), for things you run often.
- Replying to notifications through the WhatsApp tool.
