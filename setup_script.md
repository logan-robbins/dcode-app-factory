# Codex Cloud Setup Script

Paste the script below into the Codex cloud **Setup Script** field.

```bash
#!/usr/bin/env bash
set -euxo pipefail

uv --version
python --version

uv sync --all-groups --frozen
uv run pytest -q
```

## Optional Maintenance Script

Paste this into the Codex cloud **Maintenance Script** field.

```bash
#!/usr/bin/env bash
set -euxo pipefail

uv sync --all-groups --frozen
```
