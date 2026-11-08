#!/usr/bin/env bash
set -euo pipefail
set +x

if [[ -n "${TELYS_API_KEY:-}" ]]; then
    python3 - <<'PY'
import os
import tempfile

import telys.installer
import telys.login
import telys.paths

key = os.environ["TELYS_API_KEY"]
home = telys.paths.telys_home()
os.makedirs(home, mode=0o700, exist_ok=True)
license_token = telys.login.register_device(
    api=telys.paths.api_url(), api_key=key
)
license_path = os.path.join(home, "login_license.jwt")
descriptor, temporary_path = tempfile.mkstemp(dir=home, prefix=".login_license.")
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as license_file:
        license_file.write(license_token)
    os.replace(temporary_path, license_path)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)
telys.installer.install_from_host(token=key, license_path=license_path)
PY
elif [[ -n "${TELYS_OAUTH_TOKEN:-}" ]]; then
    telys login --token "$TELYS_OAUTH_TOKEN" --no-browser --plan telys_developer
fi

unset TELYS_API_KEY TELYS_OAUTH_TOKEN
exec python3 /src/main.py "$@"
