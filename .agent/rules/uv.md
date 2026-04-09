---
description: Python package management rules
---
# UV Rule

Always use `uv` instead of standard `pip` for package installation and environment management. `uv` is significantly faster.
- Use `uv pip install <package>` instead of `pip install <package>`
- Use `uv venv` to create virtual environments.
- Keep dependencies updated using `uv`.
