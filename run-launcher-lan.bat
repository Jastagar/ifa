@echo off
rem Same as run-launcher.bat, but other devices on the network can open the launcher (with the access key it prints) and use Ollama.
call "%~dp0run-launcher.bat" --lan %*
