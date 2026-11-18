"""Publish GitHub assets from an exact, reviewed draft."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

REPOSITORY = "mikkel/yue2-concept-sliders"
RELEASES = f"repos/{REPOSITORY}/releases"
TAG = re.compile(r"v[1-9][0-9]*(?:\.[0-9]+\.[0-9]+)?")
ASSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def request_from(path):
    request = json.loads(path.read_bytes())
    fields = {"tag", "release_id", "source_commit", "prerelease", "files"}
    if not isinstance(request, dict) or request.keys() != fields:
        raise ValueError("Unexpected release request fields")
    if not isinstance(request["tag"], str) or TAG.fullmatch(request["tag"]) is None:
        raise ValueError("Expected an established release tag")
    if not isinstance(request["source_commit"], str) or COMMIT.fullmatch(request["source_commit"]) is None:
        raise ValueError("Expected the full source commit")
    if type(request["release_id"]) is not int or request["release_id"] <= 0 or type(request["prerelease"]) is not bool:
        raise ValueError("Expected a numeric release ID and a prerelease boolean")
    files = request["files"]
    if not isinstance(files, dict):
        raise TypeError("Expected reviewed file records")
    for name, record in files.items():
        if (ASSET_NAME.fullmatch(name) is None or not isinstance(record, dict)
                or record.keys() != {"bytes", "sha256"} or type(record["bytes"]) is not int
                or record["bytes"] <= 0 or not isinstance(record["sha256"], str)
                or SHA256.fullmatch(record["sha256"]) is None):
            raise ValueError(f"Invalid file record for {name}")
    validate_manifest(request)
    return request


def output(*args, input=None):
    return subprocess.check_output(args, input=input, timeout=60)


def gh_json(*args, payload=None):
    if payload is None:
        return json.loads(output("gh", "api", *args))
    return json.loads(output("gh", "api", *args, "--input", "-", input=json.dumps(payload).encode()))


def verify_source(source):
    if output("git", "cat-file", "-t", source).strip() != b"commit":
        raise ValueError("Release source must be a commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], check=True)


def verify_tag(request, required=False):
    ref = f"refs/tags/{request['tag']}"
    result = output("git", "ls-remote", f"https://github.com/{REPOSITORY}.git", ref, f"{ref}^{{}}")
    refs = {}
    for line in result.decode().splitlines():
        sha, name = line.split("\t")
        refs[name] = sha
    current = refs.get(f"{ref}^{{}}", refs.get(ref))
    if (current is not None or required) and current != request["source_commit"]:
        raise ValueError("Release tag does not point to the reviewed source commit")
    return current


def verified_record(request, destination=None):
    release_id = request["release_id"]
    record = gh_json(f"{RELEASES}/{release_id}")
    if (record["id"] != release_id or record["tag_name"] != request["tag"]
            or record["prerelease"] != request["prerelease"]
            or (record["draft"] and record["target_commitish"] != request["source_commit"])):
        raise ValueError("GitHub release differs from the reviewed tag, target or release type")
    pages = output(
        "gh", "api", "--paginate", f"{RELEASES}/{release_id}/assets?per_page=100",
        "--jq", ".[] | @json",
    )
    assets = {}
    for line in pages.splitlines():
        asset = json.loads(line)
        name = asset["name"]
        spec = request["files"].get(name)
        if (name in assets or spec is None or asset["state"] != "uploaded"
                or asset["size"] != spec["bytes"] or asset["digest"] != f"sha256:{spec['sha256']}"):
            raise ValueError(f"GitHub asset differs from the reviewed file {name}")
        assets[name] = asset["id"]
    if assets.keys() != request["files"].keys():
        raise ValueError("GitHub assets do not cover the reviewed files")
    if destination is not None:
        for name, asset_id in assets.items():
            download_asset(asset_id, destination / name, request["files"][name])
        verify_payload(request, destination)
    return record


def download_asset(asset_id, path, spec):
    with path.open("wb") as file:
        subprocess.run(
            ["gh", "api", "-H", "Accept: application/octet-stream", f"{RELEASES}/assets/{asset_id}"],
            stdout=file, check=True, timeout=240,
        )
    with path.open("rb") as file:
        size, digest = path.stat().st_size, hashlib.file_digest(file, "sha256").hexdigest()
    if size != spec["bytes"] or digest != spec["sha256"]:
        raise ValueError(f"Downloaded bytes differ from the reviewed file {path.name}")


ARCHIVE_SUFFIXES = ("-particles.zip", "-native-loras.zip", "-comfyui-loras.zip")


def release_files(request):
    names = request["files"].keys()
    archives = [name for suffix in ARCHIVE_SUFFIXES for name in names if name.endswith(suffix)]
    records = [name for name in names if name.endswith("-verification.json")]
    if len(names) != 5 or len(archives) != 3 or any(
        sum(name.endswith(suffix) for name in archives) != 1 for suffix in ARCHIVE_SUFFIXES
    ) or len(records) != 1 or "SHA256SUMS" not in names:
        raise ValueError("Expected three bundle ZIPs, SHA256SUMS and one verification record")
    return archives, records[0]


def validate_manifest(request):
    release_files(request)


def archive_controls(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        files = set(names)
        if len(files) != len(names) or archive.testzip() is not None:
            raise ValueError(f"Duplicate or damaged entries in {path.name}")
        weights = {name for name in files if name.endswith(".safetensors")}
        controls = {Path(name).stem.split("_", 1)[0] for name in weights}
        if len(weights) != expected or len(controls) != expected:
            raise ValueError(f"Unexpected number of slider weights in {path.name}")
        for name in weights:
            if name + ".json" not in files and name.removesuffix(".safetensors") + ".json" not in files:
                raise ValueError(f"Missing configuration in {path.name} for {name}")
    return controls


def verify_payload(request, directory):
    archives, verification = release_files(request)
    found = {}
    for line in (directory / "SHA256SUMS").read_text("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match is None or match[2] in found:
            raise ValueError("Invalid or duplicate checksum record")
        found[match[2]] = match[1]
    if found != {name: request["files"][name]["sha256"] for name in archives}:
        raise ValueError("SHA256SUMS does not match the reviewed ZIPs")
    record = json.loads((directory / verification).read_bytes())
    distillation = record["distillation"]
    sliders = distillation["sliders"]
    if (distillation["complete"] is not True or type(sliders) is not int or sliders <= 0
            or len(distillation["reports"]) != sliders or record["particles"]["real_checkpoints"] != sliders):
        raise ValueError("Verification record does not describe the complete set of sliders")
    controls = [archive_controls(directory / name, sliders) for name in archives]
    if any(item != controls[0] for item in controls[1:]):
        raise ValueError("The three archives do not contain the same controls")


def inspected_release(request):
    with tempfile.TemporaryDirectory(prefix="github-release-") as directory:
        record = verified_record(request, Path(directory))
    current = verify_tag(request, required=not record["draft"])
    return record, current


def publish(request):
    record, current = inspected_release(request)
    if not record["draft"]:
        return record["html_url"]
    if current is None:
        gh_json(
            "--method", "POST", f"repos/{REPOSITORY}/git/refs",
            payload={"ref": f"refs/tags/{request['tag']}", "sha": request["source_commit"]},
        )
    verify_tag(request, required=True)
    gh_json(
        "--method", "PATCH", f"{RELEASES}/{request['release_id']}",
        payload={"draft": False, "prerelease": request["prerelease"],
                 "make_latest": "false" if request["prerelease"] else "legacy"},
    )
    published = verified_record(request)
    verify_tag(request, required=True)
    if published["draft"]:
        raise ValueError("The reviewed release is still a draft")
    return published["html_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validate", "audit", "publish"))
    parser.add_argument("--request", type=Path, default=Path("release.json"))
    args = parser.parse_args()
    request = request_from(args.request)
    verify_source(request["source_commit"])
    if args.operation == "audit":
        record, _ = inspected_release(request)
        print(f"Verified {record['tag_name']} and {len(request['files'])} attached files")
    if args.operation == "publish":
        if (os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
                or os.environ.get("GITHUB_EVENT_NAME") != "push"
                or os.environ.get("GITHUB_REF") != "refs/heads/main"
                or output("git", "rev-parse", "HEAD").decode().strip() != os.environ.get("GITHUB_SHA")):
            raise ValueError("Publication requires a push to the release repository's main branch")
        print(publish(request))


if __name__ == "__main__":
    main()
