#!/usr/bin/env bash
# Ifa voice-mode launcher for macOS. Double-click this file in Finder.
# Thin wrapper: sets IFA_MODE=voice then delegates to run.command.

cd "$(dirname "$0")"
export IFA_MODE=voice
exec bash run.command
