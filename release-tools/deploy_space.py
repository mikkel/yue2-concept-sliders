#!/usr/bin/env python3
"""Release the tracked YuE2 app to its existing Hugging Face Space."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.hf_api import RepoFile

ROOT = Path(__file__).resolve().parents[1]
SPACE = "ntc-ai/yue2-particle-sliders"
HOST = "https://ntc-ai-yue2-particle-sliders.hf.space"
HARDWARE = "zero-a10g"
BUILD_FAILURES = {"BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR", "NO_APP_FILE"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def payload():
    files = {}
    for path in git("ls-files", "-z", "space/").split("\0"):
        if not path:
            continue
        source = ROOT / path
        name = path.removeprefix("space/")
        if source.resolve() != source or not source.is_file() or name == ".gitattributes":
            raise ValueError(f"Invalid Space source file: {path}")
        data = source.read_bytes()
        if source.suffix == ".py":
            compile(data, path, "exec")
        files[name] = data
    if not {"README.md", "app.py", "catalog.json", "requirements.txt"} <= files.keys():
        raise ValueError("Space source is missing a required file")
    return files


def blob_id(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def remote_files(api, revision):
    return {
        item.path: item
        for item in api.list_repo_tree(SPACE, repo_type="space", revision=revision, recursive=True)
        if isinstance(item, RepoFile)
    }


def verify_examples(files):
    manifest = json.loads((ROOT / "release-tools/space-examples.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        item = files.get(name)
        if item is None or item.lfs is None or item.lfs.sha256 != digest:
            raise RuntimeError(f"Recorded Space example differs from its manifest: {name}")


def verify_running(api, revision):
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if api.space_info(SPACE).sha != revision:
            raise RuntimeError("Space head changed during verification")
        runtime = api.get_space_runtime(SPACE)
        if runtime.raw.get("sha") == revision:
            if runtime.stage in BUILD_FAILURES:
                raise RuntimeError(f"Space revision {revision} entered {runtime.stage}")
            if runtime.stage == "RUNNING" and runtime.hardware == HARDWARE:
                try:
                    with urlopen(f"{HOST}/config", timeout=15) as response:
                        config = json.load(response)
                except (URLError, TimeoutError):
                    time.sleep(10)
                    continue
                endpoints = {item.get("api_name") for item in config["dependencies"]}
                if not {"generate", "listening_pair"} <= endpoints:
                    raise RuntimeError("The running Space is missing an expected Gradio endpoint")
                print(f"Verified running Space revision {revision} and its Gradio endpoints")
                return
        time.sleep(10)
    raise RuntimeError(f"Space revision {revision} did not become healthy in one hour")


def release(files, *, apply):
    public = HfApi(endpoint="https://huggingface.co", token=False)
    info = public.space_info(SPACE)
    runtime = public.get_space_runtime(SPACE)
    if info.host != HOST or info.sdk != "gradio" or info.private or runtime.hardware != HARDWARE:
        raise RuntimeError("Unexpected Space configuration")

    original = remote_files(public, info.sha)
    verify_examples(original)
    expected = {name: blob_id(data) for name, data in files.items()}
    changed = [name for name, digest in expected.items() if name not in original or original[name].blob_id != digest]
    print(f"Space {info.sha}: {len(changed)} changed files out of {len(files)}")
    for name in changed:
        print(name)
    if not apply:
        return
    revision = info.sha
    if changed:
        source = git("rev-parse", "HEAD").strip()
        remote_main = git("ls-remote", "--exit-code", "origin", "refs/heads/main").split()[0]
        if source != os.environ["GITHUB_SHA"] or source != remote_main:
            raise RuntimeError("The tested GitHub revision is no longer main")
        token = os.environ.get("HF_SPACE_WRITE_TOKEN")
        if not token:
            raise RuntimeError("Set HF_SPACE_WRITE_TOKEN in the production environment")
        writer = HfApi(endpoint="https://huggingface.co", token=token)
        revision = writer.create_commit(
            SPACE,
            repo_type="space",
            revision="main",
            parent_commit=info.sha,
            commit_message=f"Publish GitHub revision {source}",
            operations=[CommitOperationAdd(name, files[name]) for name in changed],
        ).oid
        installed = remote_files(public, revision)
        verify_examples(installed)
        expected_tree = {name: item.blob_id for name, item in original.items()} | expected
        if any(name not in installed or installed[name].blob_id != digest for name, digest in expected_tree.items()):
            raise RuntimeError("Published Space files differ from the expected tree")
    verify_running(public, revision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="publish the tested source revision")
    release(payload(), apply=parser.parse_args().apply)
