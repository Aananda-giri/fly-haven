#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if command -v uv >/dev/null 2>&1; then
  exec uv run flywirehead run
fi
if test -x .venv/bin/python; then
  exec .venv/bin/python -m flywirehead run
fi
printf '%s\n' 'Set up this app first. See README.md for the Python installation steps.'
exit 1
