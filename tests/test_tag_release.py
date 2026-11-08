# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for the stable-version tag automation."""

import base64
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / ".github" / "scripts" / "tag_release.py"
SPEC = importlib.util.spec_from_file_location("release_tag_helper_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)

HEAD = "a" * 40
BEFORE = "b" * 40
PR_HEAD = "c" * 40
LATER_MAIN = "d" * 40
READ_CREDENTIAL = "read-credential"
WRITE_CREDENTIAL = "write-credential"
GIT_EXECUTABLE = shutil.which("git")
assert GIT_EXECUTABLE is not None


def commit_ref(sha):
    return {"object": {"type": "commit", "sha": sha}}


def check(name, conclusion="success", status="completed", check_id=1):
    return {
        "id": check_id,
        "name": name,
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
    }


def pyproject(version):
    return f'[project]\nversion = "{version}"\n'.encode()


class FakeGitHubClient(release.GitHubClient):
    def __init__(self, state, token=None):
        super().__init__("thyn-ai/telys-vllm", token or READ_CREDENTIAL)
        self.state = state
        self.posts = []

    def request(self, method, path, payload=None, allow_not_found=False):
        if method == "POST":
            assert path == "git/refs"
            self.posts.append(payload)
            tag = payload["ref"].removeprefix("refs/tags/")
            self.state["refs"][f"tags/{tag}"] = commit_ref(payload["sha"])
            return {"ref": payload["ref"], "object": commit_ref(payload["sha"])}
        assert method == "GET"
        if path.startswith("git/ref/"):
            value = self.state["refs"].get(path.removeprefix("git/ref/"))
            if value is None and allow_not_found:
                return None
            assert value is not None, path
            return value
        if path.startswith("contents/pyproject.toml?ref="):
            sha = path.removeprefix("contents/pyproject.toml?ref=")
            encoded = base64.b64encode(self.state["pyprojects"][sha]).decode()
            return {"encoding": "base64", "content": encoded[:20] + "\n" + encoded[20:]}
        if path.startswith("commits/") and "/check-runs?" in path:
            sha = path.split("/", 2)[1]
            return {"check_runs": self.state["checks"].get(sha, [])}
        if path.startswith("commits/") and path.endswith("/pulls?per_page=100"):
            return self.state["pulls"]
        if path.startswith("compare/"):
            base, head = path.removeprefix("compare/").split("...")
            return {"status": self.state["comparisons"][(base, head)]}
        raise AssertionError(f"Unexpected API request: {method} {path}")


def valid_state():
    return {
        "refs": {"heads/main": commit_ref(HEAD), "tags/v0.1.3": None},
        "pyprojects": {HEAD: pyproject("0.1.3")},
        "checks": {
            HEAD: [check("guard")],
            PR_HEAD: [check("guard"), check("codna review", "neutral")],
        },
        "pulls": [
            {
                "merged_at": "2026-11-08T00:00:00Z",
                "merge_commit_sha": HEAD,
                "base": {"ref": "main"},
                "head": {"sha": PR_HEAD},
            }
        ],
        "comparisons": {},
    }


def plan():
    return release.ReleasePlan(before=BEFORE, head=HEAD, version="0.1.3")


def evaluate(state):
    return release.evaluate(plan(), FakeGitHubClient(state), max_wait_seconds=0)


def test_stable_version_change_and_rejections():
    assert release.changed_stable_version("0.1.3", "0.1.2") == "0.1.3"
    assert release.changed_stable_version("0.1.2", "0.1.2") is None
    with pytest.raises(release.ReleaseTagError, match="downgrade"):
        release.changed_stable_version("0.1.2", "0.1.3")
    for invalid in ("0.1.3rc1", "01.2.3", "1.2", "1.2.3.4"):
        with pytest.raises(release.ReleaseTagError, match="stable X.Y.Z"):
            release.parse_stable_version(pyproject(invalid), "test")


def test_build_plan_reads_exact_before_and_head(tmp_path):
    def git(*arguments):
        return subprocess.check_output(  # noqa: S603
            [GIT_EXECUTABLE, "-C", str(tmp_path), *arguments], text=True
        ).strip()

    subprocess.run(  # noqa: S603
        [GIT_EXECUTABLE, "-C", str(tmp_path), "init", "-q"], check=True
    )
    git("config", "user.name", "Release Test")
    git("config", "user.email", "release@example.invalid")
    (tmp_path / "pyproject.toml").write_bytes(pyproject("0.1.2"))
    git("add", "pyproject.toml")
    git("commit", "-qm", "before")
    before = git("rev-parse", "HEAD")
    (tmp_path / "pyproject.toml").write_bytes(pyproject("0.1.3"))
    git("add", "pyproject.toml")
    git("commit", "-qm", "release")
    head = git("rev-parse", "HEAD")

    result = release.build_plan(tmp_path, before, head)
    assert result == release.ReleasePlan(before=before, head=head, version="0.1.3")


@pytest.mark.parametrize("conclusion", ["success", "neutral", "skipped"])
def test_valid_release_accepts_completed_non_failing_codna(conclusion):
    state = valid_state()
    state["checks"][PR_HEAD][1] = check("codna review", conclusion)
    result = evaluate(state)
    assert result.ready is True


def test_verify_path_does_not_create_a_tag(monkeypatch):
    state = valid_state()
    reader = FakeGitHubClient(state)
    monkeypatch.setattr(release, "build_plan", lambda *_args: plan())
    monkeypatch.setattr(release, "GitHubClient", lambda *_args: reader)
    monkeypatch.setenv("GH_READ_TOKEN", READ_CREDENTIAL)
    monkeypatch.delenv("GH_WRITE_TOKEN", raising=False)
    emitted = {}
    monkeypatch.setattr(release, "emit", emitted.update)

    assert release.main(["verify", "--max-wait-seconds", "0"]) == 0
    assert emitted["ready"] == "true"
    assert state["refs"]["tags/v0.1.3"] is None
    assert reader.posts == []


def test_create_path_revalidates_and_creates_exact_lightweight_ref(monkeypatch):
    state = valid_state()
    reader = FakeGitHubClient(state)
    writer = FakeGitHubClient(state, WRITE_CREDENTIAL)
    clients = {READ_CREDENTIAL: reader, WRITE_CREDENTIAL: writer}
    monkeypatch.setattr(release, "build_plan", lambda *_args: plan())
    monkeypatch.setattr(release, "GitHubClient", lambda _repo, token, _url: clients[token])
    monkeypatch.setenv("GH_READ_TOKEN", READ_CREDENTIAL)
    monkeypatch.setenv("GH_WRITE_TOKEN", WRITE_CREDENTIAL)
    monkeypatch.setattr(release, "emit", lambda _outputs: None)

    assert release.main(["create", "--max-wait-seconds", "0"]) == 0
    assert writer.posts == [{"ref": "refs/tags/v0.1.3", "sha": HEAD}]
    assert reader.posts == []
    assert state["refs"]["tags/v0.1.3"] == commit_ref(HEAD)


def test_failed_guard_blocks_before_write_client_is_used(monkeypatch):
    state = valid_state()
    state["checks"][HEAD] = [check("guard", "failure")]
    reader = FakeGitHubClient(state)
    tokens_used = []

    def client_factory(_repository, token, _api_url):
        tokens_used.append(token)
        assert token == READ_CREDENTIAL
        return reader

    monkeypatch.setattr(release, "build_plan", lambda *_args: plan())
    monkeypatch.setattr(release, "GitHubClient", client_factory)
    monkeypatch.setenv("GH_READ_TOKEN", READ_CREDENTIAL)
    monkeypatch.setenv("GH_WRITE_TOKEN", WRITE_CREDENTIAL)
    monkeypatch.setattr(release, "emit", lambda _outputs: None)

    assert release.main(["create", "--max-wait-seconds", "0"]) == 1
    assert tokens_used == [READ_CREDENTIAL]
    assert state["refs"]["tags/v0.1.3"] is None


def test_pending_check_times_out_without_tagging():
    state = valid_state()
    state["checks"][HEAD] = [check("guard", status="in_progress")]
    with pytest.raises(release.ReleaseTagError, match="Timed out"):
        evaluate(state)
    assert state["refs"]["tags/v0.1.3"] is None


def test_direct_push_without_merged_pr_is_rejected():
    state = valid_state()
    state["pulls"] = []
    with pytest.raises(release.ReleaseTagError, match="Direct pushes"):
        evaluate(state)


def test_existing_identical_tag_is_idempotent_but_conflict_is_rejected():
    state = valid_state()
    state["refs"]["tags/v0.1.3"] = commit_ref(HEAD)
    with pytest.raises(release.ReleaseNotNeededError, match="already points"):
        evaluate(state)

    state["refs"]["tags/v0.1.3"] = commit_ref(LATER_MAIN)
    with pytest.raises(release.ReleaseTagError, match="refusing to re-tag"):
        evaluate(state)


def test_main_can_advance_with_same_version():
    state = valid_state()
    state["refs"]["heads/main"] = commit_ref(LATER_MAIN)
    state["pyprojects"][LATER_MAIN] = pyproject("0.1.3")
    state["comparisons"][(HEAD, LATER_MAIN)] = "ahead"
    assert evaluate(state).ready is True


def test_newer_main_version_supersedes_release():
    state = valid_state()
    state["refs"]["heads/main"] = commit_ref(LATER_MAIN)
    state["pyprojects"][LATER_MAIN] = pyproject("0.1.4")
    state["comparisons"][(HEAD, LATER_MAIN)] = "ahead"
    with pytest.raises(release.ReleaseNotNeededError, match="superseded"):
        evaluate(state)


def test_diverged_main_is_rejected():
    state = valid_state()
    state["refs"]["heads/main"] = commit_ref(LATER_MAIN)
    state["pyprojects"][LATER_MAIN] = pyproject("0.1.3")
    state["comparisons"][(HEAD, LATER_MAIN)] = "diverged"
    with pytest.raises(release.ReleaseTagError, match="not an ancestor"):
        evaluate(state)


def test_latest_duplicate_check_controls_the_result():
    runs = [check("guard", "failure", check_id=1), check("guard", "success", check_id=2)]
    assert release.latest_check(runs, "guard")["conclusion"] == "success"
