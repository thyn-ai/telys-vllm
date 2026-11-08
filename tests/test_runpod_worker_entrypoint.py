# SPDX-License-Identifier: Apache-2.0
"""Deterministic checks for the Runpod worker entrypoint."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ENTRYPOINT = (
    Path(__file__).resolve().parent.parent
    / "integrations"
    / "runpod-worker-vllm"
    / "entrypoint.sh"
)
OAUTH_TOKEN = "oauth-test-token"
API_KEY = "de_test-api-key"


def _write_source(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


def _write_executable(path: Path, source: str) -> None:
    _write_source(path, source)
    path.chmod(0o755)


def _make_runtime(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    calls = tmp_path / "calls"
    calls.mkdir()
    sdk_dir = tmp_path / "sdk" / "telys"
    _write_source(sdk_dir / "__init__.py", "")
    _write_source(
        sdk_dir / "paths.py",
        """
        import os

        def telys_home():
            return os.path.abspath(
                os.environ.get("TELYS_HOME")
                or os.path.join(os.path.expanduser("~"), ".telys")
            )

        def api_url():
            return "https://api.test"
        """,
    )
    _write_source(
        sdk_dir / "login.py",
        """
        import json
        import os

        def register_device(*, api, api_key):
            record_path = os.path.join(os.environ["CALL_DIR"], "register-device.json")
            with open(record_path, "w", encoding="utf-8") as record_file:
                json.dump({"api": api, "api_key": api_key}, record_file)
            if os.environ.get("TELYS_STUB_PROVISION_FAILURE") == "register":
                raise RuntimeError("register_device failed")
            return "license-test-token"
        """,
    )
    _write_source(
        sdk_dir / "installer.py",
        """
        import json
        import os

        def install_from_host(*, token=None, license_path=None, **kwargs):
            with open(license_path, encoding="utf-8") as license_file:
                license_token = license_file.read()
            record_path = os.path.join(os.environ["CALL_DIR"], "installer.json")
            with open(record_path, "w", encoding="utf-8") as record_file:
                json.dump(
                    {
                        "token": token,
                        "license_path": license_path,
                        "license_token": license_token,
                    },
                    record_file,
                )
            if os.environ.get("TELYS_STUB_PROVISION_FAILURE") == "install":
                raise RuntimeError("install_from_host failed")
            return {"installed": True}
        """,
    )
    _write_executable(
        bin_dir / "telys",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        : > "$CALL_DIR/telys-called"
        printf '%s\\n' "$@" > "$CALL_DIR/telys-args"
        exit "${TELYS_STUB_LOGIN_STATUS:-0}"
        """,
    )
    _write_executable(
        bin_dir / "python3",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${1:-}" == "-" ]]; then
            : > "$CALL_DIR/provision-called"
            exec "$REAL_PYTHON3" "$@"
        fi
        : > "$CALL_DIR/worker-called"
        printf '%s\\n' "$@" > "$CALL_DIR/worker-args"
        if [[ -z "${TELYS_OAUTH_TOKEN+x}" ]]; then
            printf 'unset\\n' > "$CALL_DIR/oauth-token-state"
        else
            printf 'set\\n' > "$CALL_DIR/oauth-token-state"
        fi
        if [[ -z "${TELYS_API_KEY+x}" ]]; then
            printf 'unset\\n' > "$CALL_DIR/api-key-state"
        else
            printf 'set\\n' > "$CALL_DIR/api-key-state"
        fi
        """,
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
    env["CALL_DIR"] = str(calls)
    env["REAL_PYTHON3"] = sys.executable
    env["HOME"] = str(tmp_path / "home")
    env["TELYS_HOME"] = str(tmp_path / "telys-home")
    pythonpath = [str(tmp_path / "sdk")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.pop("TELYS_OAUTH_TOKEN", None)
    env.pop("TELYS_API_KEY", None)
    env.pop("TELYS_STUB_LOGIN_STATUS", None)
    env.pop("TELYS_STUB_PROVISION_FAILURE", None)
    return env, calls


def _run_entrypoint(
    env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ENTRYPOINT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=10,
    )


def test_no_credentials_skip_login_and_start_preinstalled_runtime(tmp_path):
    env, calls = _make_runtime(tmp_path)

    result = _run_entrypoint(env)

    assert result.returncode == 0, result.stderr
    assert not (calls / "provision-called").exists()
    assert not (calls / "telys-called").exists()
    assert (calls / "worker-called").exists()
    assert (calls / "worker-args").read_text().splitlines() == ["/src/main.py"]


def test_oauth_login_uses_exact_headless_arguments(tmp_path):
    env, calls = _make_runtime(tmp_path)
    env["TELYS_OAUTH_TOKEN"] = OAUTH_TOKEN

    result = _run_entrypoint(env)

    assert result.returncode == 0, result.stderr
    assert not (calls / "provision-called").exists()
    assert (calls / "telys-args").read_text().splitlines() == [
        "login",
        "--token",
        OAUTH_TOKEN,
        "--no-browser",
        "--plan",
        "telys_developer",
    ]
    assert OAUTH_TOKEN not in result.stdout + result.stderr


def test_oauth_token_is_unset_before_worker_exec(tmp_path):
    env, calls = _make_runtime(tmp_path)
    env["TELYS_OAUTH_TOKEN"] = OAUTH_TOKEN

    result = _run_entrypoint(env)

    assert result.returncode == 0, result.stderr
    assert (calls / "oauth-token-state").read_text().strip() == "unset"


def test_worker_arguments_are_forwarded_verbatim(tmp_path):
    env, calls = _make_runtime(tmp_path)

    result = _run_entrypoint(env, "--worker-arg", "value with spaces")

    assert result.returncode == 0, result.stderr
    assert (calls / "worker-args").read_text().splitlines() == [
        "/src/main.py",
        "--worker-arg",
        "value with spaces",
    ]


def test_login_failure_does_not_invoke_worker(tmp_path):
    env, calls = _make_runtime(tmp_path)
    env["TELYS_OAUTH_TOKEN"] = OAUTH_TOKEN
    env["TELYS_STUB_LOGIN_STATUS"] = "42"

    result = _run_entrypoint(env)

    assert result.returncode == 42
    assert (calls / "telys-called").exists()
    assert not (calls / "worker-called").exists()
    assert OAUTH_TOKEN not in result.stdout + result.stderr


def test_api_key_direct_provisioning_uses_sdk_and_writes_license(tmp_path):
    env, calls = _make_runtime(tmp_path)
    env["TELYS_API_KEY"] = API_KEY

    result = _run_entrypoint(env)

    assert result.returncode == 0, result.stderr
    assert (calls / "provision-called").exists()
    assert not (calls / "telys-called").exists()
    assert json.loads((calls / "register-device.json").read_text()) == {
        "api": "https://api.test",
        "api_key": API_KEY,
    }
    license_path = Path(env["TELYS_HOME"]) / "login_license.jwt"
    assert license_path.read_text() == "license-test-token"
    assert license_path.stat().st_mode & 0o777 == 0o600
    assert json.loads((calls / "installer.json").read_text()) == {
        "token": API_KEY,
        "license_path": str(license_path),
        "license_token": "license-test-token",
    }
    assert (calls / "worker-called").exists()
    assert API_KEY not in result.stdout + result.stderr


def test_api_key_is_unset_before_worker_exec(tmp_path):
    env, calls = _make_runtime(tmp_path)
    env["TELYS_API_KEY"] = API_KEY

    result = _run_entrypoint(env)

    assert result.returncode == 0, result.stderr
    assert (calls / "api-key-state").read_text().strip() == "unset"


def test_api_key_failure_never_invokes_oauth_login_or_worker(tmp_path):
    for stage in ("register", "install"):
        case_dir = tmp_path / stage
        case_dir.mkdir()
        env, calls = _make_runtime(case_dir)
        env["TELYS_API_KEY"] = API_KEY
        env["TELYS_OAUTH_TOKEN"] = OAUTH_TOKEN
        env["TELYS_STUB_PROVISION_FAILURE"] = stage

        result = _run_entrypoint(env)

        assert result.returncode == 1, stage
        assert (calls / "provision-called").exists(), stage
        assert not (calls / "telys-called").exists(), stage
        assert not (calls / "worker-called").exists(), stage
        assert API_KEY not in result.stdout + result.stderr, stage
        assert OAUTH_TOKEN not in result.stdout + result.stderr, stage
