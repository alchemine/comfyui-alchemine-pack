# -*- coding: utf-8 -*-
"""Fetch the pack's data files on first use.

The statistics tables are tens of megabytes and are rebuilt on their own
schedule, so committing them would grow the repository by their full
size on every rebuild. They ship as GitHub release assets instead and
land in resources/ the first time a node needs one.

Each artifact is pinned by sha256: bump the release tag and the digest
together, never one alone, or an old cached copy will be accepted as the
new one.
"""
import os

RELEASE = ("https://github.com/alchemine/comfyui-alchemine-pack"
           "/releases/download/%s/%s")


def url_for(tag, filename):
    return RELEASE % (tag, filename)


def ensure(path, url, sha256, label, note=""):
    """Return `path`, downloading and verifying it if it is missing.

    Downloads land on a .part file and are renamed only after the digest
    matches, so an interrupted fetch can never masquerade as the real
    artifact.
    """
    if os.path.exists(path):
        return path

    import hashlib
    import urllib.request

    print("[%s] downloading %s%s from %s"
          % (label, os.path.basename(path), note and " (%s)" % note, url))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(
                "%s checksum mismatch (corrupt or stale download)"
                % os.path.basename(path))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print("[%s] %s ready" % (label, os.path.basename(path)))
    return path
