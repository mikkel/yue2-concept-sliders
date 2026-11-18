import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

script = Path(__file__).with_name("publish_github_release.py")
spec = importlib.util.spec_from_file_location("approved_release", script)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def file_record(path):
    with path.open("rb") as file:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(file, "sha256").hexdigest()}


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.source = "a" * 40
        self.request = {"tag": "v3", "release_id": 42, "source_commit": self.source, "prerelease": False}
        self.draft = {"draft": True, "html_url": "https://example.com/release"}
        names = [f"yue2-v3{suffix}" for suffix in release.ARCHIVE_SUFFIXES]
        for name in names:
            with zipfile.ZipFile(self.path / name, "w") as archive:
                archive.writestr("example_step1600.safetensors", b"weight fixture")
                archive.writestr("example_step1600.json", "{}")
        (self.path / "SHA256SUMS").write_text("".join(
            f"{file_record(self.path / name)['sha256']}  {name}\n" for name in names
        ))
        (self.path / "v3-verification.json").write_text(json.dumps({
            "distillation": {"complete": True, "sliders": 1, "reports": ["example"]},
            "particles": {"real_checkpoints": 1},
        }))
        self.request["files"] = {file.name: file_record(file) for file in self.path.iterdir()}

    def test_request_matches_reviewed_shape(self):
        request = self.path / "release.json"
        request.write_text(json.dumps(self.request))
        self.assertEqual(release.request_from(request), self.request)

    def test_publish_creates_checked_tag_and_rechecks_release(self):
        with (
            patch.object(release, "inspected_release", return_value=(self.draft, None)),
            patch.object(release, "verified_record", return_value=dict(self.draft, draft=False)),
            patch.object(release, "verify_tag") as ref,
            patch.object(release, "gh_json") as api,
        ):
            self.assertEqual(release.publish(self.request), self.draft["html_url"])
        create_tag, publish = api.call_args_list
        self.assertEqual(create_tag.args[1], "POST")
        self.assertEqual(create_tag.kwargs["payload"]["sha"], self.source)
        self.assertEqual(publish.args[1], "PATCH")
        self.assertFalse(publish.kwargs["payload"]["draft"])
        self.assertEqual(ref.call_count, 2)

    def test_existing_release_is_idempotent(self):
        current = dict(self.draft, draft=False)
        with (
            patch.object(release, "inspected_release", return_value=(current, self.source)),
            patch.object(release, "gh_json") as api,
        ):
            self.assertEqual(release.publish(self.request), current["html_url"])
        api.assert_not_called()

    def test_invalid_payload_never_reaches_a_write(self):
        with (
            patch.object(release, "inspected_release", side_effect=ValueError("Invalid payload")),
            patch.object(release, "gh_json") as api,
            self.assertRaisesRegex(ValueError, "Invalid payload"),
        ):
            release.publish(self.request)
        api.assert_not_called()

    def test_verifies_all_archive_roles_and_checksums(self):
        release.verify_payload(self.request, self.path)

    def test_refuses_unreviewed_checksum(self):
        (self.path / "SHA256SUMS").write_text("0" * 64 + "  yue2-v3-particles.zip\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            release.verify_payload(self.request, self.path)

    def test_requires_each_slider_configuration(self):
        path = self.path / "incomplete.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("example_step1600.safetensors", b"data")
        with self.assertRaisesRegex(ValueError, "Missing configuration"):
            release.archive_controls(path, 1)
