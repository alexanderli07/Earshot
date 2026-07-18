"""Checksum-verified, atomic downloads for Earshot runtime artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30.0
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_INSTALL_LOCK = threading.Lock()


@dataclass(frozen=True)
class Artifact:
    """A remote artifact and the digest required before installation."""

    url: str
    path: Path
    sha256: str


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be downloaded or installed."""


class ChecksumError(ArtifactError):
    """Raised when a downloaded artifact does not match its expected digest."""


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_artifact(artifact: Artifact) -> bool:
    """Download and atomically install *artifact* if it is not already valid.

    Returns ``True`` when a download occurred and ``False`` when the existing
    destination already matched the expected digest.
    """

    destination = artifact.path
    part: Path | None = None

    try:
        with _INSTALL_LOCK:
            if (destination.is_file()
                    and sha256_file(destination) == artifact.sha256):
                return False

        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(artifact.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                declared_size = int(content_length)
                if declared_size < 0:
                    raise ValueError(
                        f"Invalid negative Content-Length for {destination.name}"
                    )
                if declared_size > MAX_ARTIFACT_BYTES:
                    raise ValueError(
                        f"Artifact exceeds the {MAX_ARTIFACT_BYTES}-byte maximum "
                        f"({declared_size} bytes declared)"
                    )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".part",
                delete=False,
            ) as output:
                part = Path(output.name)
                transferred = 0
                while chunk := response.read(_CHUNK_SIZE):
                    transferred += len(chunk)
                    if transferred > MAX_ARTIFACT_BYTES:
                        raise ValueError(
                            f"Artifact exceeds the {MAX_ARTIFACT_BYTES}-byte maximum "
                            "while streaming"
                        )
                    output.write(chunk)

        actual = sha256_file(part)
        if actual != artifact.sha256:
            raise ChecksumError(
                f"Checksum mismatch for {destination.name}: "
                f"expected {artifact.sha256}, got {actual}"
            )

        # Downloads may overlap, but Windows can reject destination reads and
        # replacements that overlap. The same short lock covers the initial
        # validation above and this winner check/install boundary.
        with _INSTALL_LOCK:
            if (destination.is_file()
                    and sha256_file(destination) == artifact.sha256):
                return True
            os.replace(part, destination)
        return True
    except ChecksumError:
        raise
    except Exception as exc:
        raise ArtifactError(
            f"Could not install {destination.name} from {artifact.url}: {exc}"
        ) from exc
    finally:
        if part is not None:
            try:
                part.unlink()
            except OSError:
                pass
