# SPDX-License-Identifier: Apache-2.0
"""Validate a main-branch version bump and optionally create its immutable tag."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time

import tomllib  # isort: skip
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STABLE_VERSION_RE = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
SHA_RE = re.compile(r"[0-9a-f]{40}")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
NON_FAILING_CONCLUSIONS = {"success", "neutral", "skipped"}
DEFAULT_MAX_WAIT_SECONDS = 1800
DEFAULT_POLL_SECONDS = 15


class ReleaseTagError(RuntimeError):
    """A validation or API failure that must prevent tag creation."""


class ReleaseNotNeededError(RuntimeError):
    """A valid state in which no new tag should be created."""


@dataclass(frozen=True)
class ReleasePlan:
    before: str
    head: str
    version: str

    @property
    def tag(self) -> str:
        return f"v{self.version}"


@dataclass(frozen=True)
class Evaluation:
    ready: bool
    reason: str


def version_tuple(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def parse_stable_version(document: bytes, source: str) -> str:
    try:
        project = tomllib.loads(document.decode("utf-8"))["project"]
        version = project["version"]
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseTagError(f"Could not read project.version from {source}: {exc}") from exc
    if not isinstance(version, str) or not STABLE_VERSION_RE.fullmatch(version):
        raise ReleaseTagError(
            f"project.version in {source} must be stable X.Y.Z without leading zeroes; "
            f"received {version!r}"
        )
    return version


def changed_stable_version(current: str, previous: str) -> str | None:
    if current == previous:
        return None
    if version_tuple(current) < version_tuple(previous):
        raise ReleaseTagError(f"Version downgrade on main is forbidden: {previous} -> {current}")
    return current


def git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise ReleaseTagError("git is required but was not found on PATH")
    return executable


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(  # noqa: S603, RUF100
        [git_executable(), "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ReleaseTagError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def validate_sha(value: str, label: str) -> None:
    if not SHA_RE.fullmatch(value):
        raise ReleaseTagError(f"{label} must be a full lowercase Git SHA; received {value!r}")


def build_plan(repository: Path, before: str, head: str) -> ReleasePlan | None:
    validate_sha(before, "before SHA")
    validate_sha(head, "head SHA")
    actual_head = run_git(repository, "rev-parse", "HEAD")
    if actual_head != head:
        raise ReleaseTagError(f"Checked-out HEAD {actual_head} does not match event SHA {head}")
    run_git(repository, "rev-parse", "--verify", f"{before}^{{commit}}")
    run_git(repository, "rev-parse", "--verify", f"{head}^{{commit}}")
    ancestor = subprocess.run(  # noqa: S603, RUF100
        [git_executable(), "-C", str(repository), "merge-base", "--is-ancestor", before, head],
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ReleaseTagError("The push's before SHA is not an ancestor of its head SHA")
    current = parse_stable_version(
        (repository / "pyproject.toml").read_bytes(), f"pyproject.toml at {head}"
    )
    previous_document = run_git(repository, "show", f"{before}:pyproject.toml")
    previous = parse_stable_version(
        previous_document.encode("utf-8"), f"pyproject.toml at {before}"
    )
    version = changed_stable_version(current, previous)
    return ReleasePlan(before=before, head=head, version=version) if version else None


class GitHubClient:
    def __init__(
        self, repository: str, token: str, api_url: str = "https://api.github.com"
    ) -> None:
        if not token:
            raise ReleaseTagError("A GitHub API token is required")
        if not REPOSITORY_RE.fullmatch(repository):
            raise ReleaseTagError(f"Invalid GitHub repository: {repository!r}")
        parsed_api = urllib.parse.urlparse(api_url)
        if parsed_api.scheme != "https" or not parsed_api.netloc:
            raise ReleaseTagError("GitHub API URL must be an absolute HTTPS URL")
        self.repository = repository
        self.token = token
        self.api_url = api_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        url = f"{self.api_url}/repos/{self.repository}/{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(  # noqa: S310, RUF100
            url,
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "telys-vllm-release-tag",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310, RUF100
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ReleaseTagError(
                f"GitHub API {method} {path} returned {exc.code}: {detail}"
            ) from exc
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ReleaseTagError(f"GitHub API {method} {path} failed: {exc}") from exc

    def get_ref(self, ref: str) -> dict[str, Any] | None:
        quoted = urllib.parse.quote(ref, safe="/")
        return self.request("GET", f"git/ref/{quoted}", allow_not_found=True)

    def get_pyproject(self, sha: str) -> bytes:
        result = self.request("GET", f"contents/pyproject.toml?ref={sha}")
        if result.get("encoding") != "base64" or not isinstance(result.get("content"), str):
            raise ReleaseTagError(f"Unexpected pyproject.toml content response at {sha}")
        try:
            encoded = "".join(result["content"].split())
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ReleaseTagError(
                f"Invalid pyproject.toml content encoding at {sha}: {exc}"
            ) from exc

    def get_check_runs(self, sha: str) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        page = 1
        while True:
            result = self.request(
                "GET", f"commits/{sha}/check-runs?per_page=100&page={page}"
            )
            batch = result.get("check_runs", [])
            runs.extend(batch)
            if len(batch) < 100:
                return runs
            page += 1

    def get_merged_pull(self, sha: str) -> dict[str, Any]:
        pulls = self.request("GET", f"commits/{sha}/pulls?per_page=100")
        matches = [
            pull
            for pull in pulls
            if pull.get("merged_at")
            and pull.get("merge_commit_sha") == sha
            and pull.get("base", {}).get("ref") == "main"
        ]
        if len(matches) != 1:
            raise ReleaseTagError(
                "Version bump must have exactly one associated PR merged into main at the "
                f"event commit; found {len(matches)}. Direct pushes are not tagged."
            )
        return matches[0]

    def compare(self, base: str, head: str) -> str:
        return self.request("GET", f"compare/{base}...{head}")["status"]

    def create_tag(self, plan: ReleasePlan) -> None:
        self.request(
            "POST",
            "git/refs",
            payload={"ref": f"refs/tags/{plan.tag}", "sha": plan.head},
        )


def latest_check(runs: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [run for run in runs if run.get("name") == name]
    return max(matches, key=lambda run: run.get("id", 0)) if matches else None


def wait_for_check(
    client: GitHubClient,
    sha: str,
    name: str,
    allowed_conclusions: set[str],
    max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
) -> None:
    started = time.monotonic()
    while True:
        check = latest_check(client.get_check_runs(sha), name)
        if check and check.get("status") == "completed":
            conclusion = check.get("conclusion")
            if conclusion in allowed_conclusions:
                return
            raise ReleaseTagError(
                f"Required check {name!r} at {sha} concluded {conclusion!r}; "
                f"expected one of {sorted(allowed_conclusions)}"
            )
        if time.monotonic() - started >= max_wait_seconds:
            raise ReleaseTagError(
                f"Timed out waiting for required check {name!r} at {sha} to complete"
            )
        time.sleep(poll_seconds)


def validate_main_lineage(plan: ReleasePlan, client: GitHubClient) -> None:
    main_ref = client.get_ref("heads/main")
    if not main_ref or main_ref.get("object", {}).get("type") != "commit":
        raise ReleaseTagError("Could not resolve refs/heads/main to a commit")
    live_main = main_ref["object"]["sha"]
    validate_sha(live_main, "live main SHA")
    if live_main != plan.head and client.compare(plan.head, live_main) != "ahead":
        raise ReleaseTagError("The version-bump commit is not an ancestor of current main")
    live_version = parse_stable_version(
        client.get_pyproject(live_main), f"pyproject.toml on live main {live_main}"
    )
    if version_tuple(live_version) > version_tuple(plan.version):
        raise ReleaseNotNeededError(
            f"Main already has newer version {live_version}; not tagging superseded {plan.tag}"
        )
    if live_version != plan.version:
        raise ReleaseTagError(
            f"Live main version {live_version} does not match release {plan.version}"
        )


def validate_absent_or_existing_tag(plan: ReleasePlan, client: GitHubClient) -> None:
    existing = client.get_ref(f"tags/{plan.tag}")
    if not existing:
        return
    tag_object = existing.get("object", {})
    if tag_object.get("type") == "commit" and tag_object.get("sha") == plan.head:
        raise ReleaseNotNeededError(
            f"Tag {plan.tag} already points to {plan.head}; nothing to do"
        )
    raise ReleaseTagError(
        f"Tag {plan.tag} already exists at {tag_object.get('sha')!r}; refusing to re-tag"
    )


def evaluate(
    plan: ReleasePlan,
    client: GitHubClient,
    max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
) -> Evaluation:
    validate_absent_or_existing_tag(plan, client)
    validate_main_lineage(plan, client)
    pull = client.get_merged_pull(plan.head)
    pull_head = pull.get("head", {}).get("sha")
    if not isinstance(pull_head, str):
        raise ReleaseTagError("Merged pull request did not include a head SHA")
    validate_sha(pull_head, "pull request head SHA")
    wait_for_check(client, plan.head, "guard", {"success"}, max_wait_seconds, poll_seconds)
    wait_for_check(client, pull_head, "guard", {"success"}, max_wait_seconds, poll_seconds)
    wait_for_check(
        client,
        pull_head,
        "codna review",
        NON_FAILING_CONCLUSIONS,
        max_wait_seconds,
        poll_seconds,
    )
    validate_absent_or_existing_tag(plan, client)
    validate_main_lineage(plan, client)
    return Evaluation(ready=True, reason=f"Validated {plan.tag} at {plan.head}")


def emit(outputs: dict[str, str]) -> None:
    sys.stdout.write(json.dumps(outputs, sort_keys=True) + "\n")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as stream:
            stream.writelines(f"{key}={value}\n" for key, value in outputs.items())


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    parser.add_argument("--before", default=os.environ.get("BEFORE_SHA", ""))
    parser.add_argument("--head", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-wait-seconds", type=int, default=DEFAULT_MAX_WAIT_SECONDS)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "verify", "create"))
    common_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])
    try:
        plan = build_plan(args.repo_root, args.before, args.head)
        if plan is None:
            emit({"changed": "false", "ready": "false", "reason": "No stable version bump"})
            return 0
        base_outputs = {
            "changed": "true",
            "version": plan.version,
            "tag": plan.tag,
            "head": plan.head,
        }
        if args.command == "inspect":
            emit(base_outputs)
            return 0
        reader = GitHubClient(
            args.repository, os.environ.get("GH_READ_TOKEN", ""), args.api_url
        )
        try:
            evaluation = evaluate(plan, reader, args.max_wait_seconds, args.poll_seconds)
        except ReleaseNotNeededError as exc:
            emit({**base_outputs, "ready": "false", "reason": str(exc)})
            return 0
        outputs = {
            **base_outputs,
            "ready": str(evaluation.ready).lower(),
            "reason": evaluation.reason,
        }
        if args.command == "create":
            writer = GitHubClient(
                args.repository, os.environ.get("GH_WRITE_TOKEN", ""), args.api_url
            )
            writer.create_tag(plan)
            created = reader.get_ref(f"tags/{plan.tag}")
            if not created or created.get("object", {}).get("sha") != plan.head:
                raise ReleaseTagError(
                    f"Tag {plan.tag} was not readable at {plan.head} after creation"
                )
            outputs["created"] = "true"
        emit(outputs)
        return 0
    except (ReleaseTagError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"::error::{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
