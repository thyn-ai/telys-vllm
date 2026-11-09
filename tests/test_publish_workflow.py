# SPDX-License-Identifier: Apache-2.0
"""Structural regressions for the tag-only publication workflow."""

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish.yml"
).read_text(encoding="utf-8")


def test_release_job_checks_out_source_before_release_creation():
    release_job = WORKFLOW.split("\n  release:\n", 1)[1]
    checkout = release_job.index("actions/checkout@")
    create = release_job.index("name: Create or update the release")
    assert checkout < create


def test_image_digest_comes_from_registry_metadata():
    verification = WORKFLOW.split("name: Verify the published image", 1)[1].split(
        "name: Upload immutable release reference", 1
    )[0]
    assert 'imagetools inspect "${tags[0]}"' in verification
    assert "imagetools inspect --raw" not in verification


def test_attestation_flags_do_not_conflict():
    command = WORKFLOW.split("gh attestation verify", 1)[1].split("--format json", 1)[0]
    assert "--cert-identity" not in command
    assert "--signer-workflow" in command
