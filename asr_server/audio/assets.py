from __future__ import annotations

import hashlib
import wave
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from asr_server.adapters.base import AudioPath
from asr_server.errors import AsrError


NORMALIZATION_VERSION = "pcm16-wav-v2"


@dataclass(frozen=True)
class AudioSpec:
    """An engine's input format; PCM16 is the current decoder output codec."""

    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0 or self.sample_width != 2:
            raise ValueError("audio specification requires positive rate/channels and PCM16 samples")

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.sample_width


DEFAULT_AUDIO_SPEC = AudioSpec()


@dataclass(frozen=True)
class NormalizedAudioAsset:
    """Internally verified, borrowed audio. Only its workspace owner deletes it.

    Public upload parameters never construct this object. Persistent metadata
    must be verified against the actual file before it is reused after restart.
    The digest covers PCM, so an unrelated container header is never confused
    with the source upload's digest or a prior checkpoint file fingerprint.
    """

    path: Path
    spec: AudioSpec
    frame_count: int
    pcm_sha256: str
    normalization_version: str = NORMALIZATION_VERSION
    owner: str = "workspace"
    _file_identity: tuple[int, int, int, int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_file_identity", self._stat_identity())

    def _stat_identity(self) -> tuple[int, int, int, int, int]:
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @property
    def duration(self) -> float:
        return self.frame_count / self.spec.sample_rate

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def audio_view(self, start: float = 0.0, end: float | None = None) -> AudioPath:
        return AudioPath(self.path, start, end)

    def relocated(self, path: Path) -> NormalizedAudioAsset:
        """Transfer a successful atomic rename; no file is copied or deleted."""
        return replace(self, path=path)

    def to_metadata(self) -> dict[str, Any]:
        return {"spec": asdict(self.spec), "frame_count": self.frame_count,
                "pcm_sha256": self.pcm_sha256, "normalization_version": self.normalization_version,
                "owner": self.owner}

    @classmethod
    def from_path(cls, path: Path, *, spec: AudioSpec = DEFAULT_AUDIO_SPEC,
                  owner: str = "workspace") -> NormalizedAudioAsset:
        try:
            with wave.open(str(path), "rb") as source:
                actual = AudioSpec(source.getframerate(), source.getnchannels(), source.getsampwidth())
                frames = source.getnframes()
                if actual != spec or source.getcomptype() != "NONE":
                    raise ValueError("audio format differs from required specification")
                digest = hashlib.sha256()
                read_frames = 0
                while pcm := source.readframes(32_768):
                    digest.update(pcm)
                    read_frames += len(pcm) // spec.frame_bytes
                if read_frames != frames:
                    raise ValueError("audio file has incomplete PCM data")
                return cls(path, spec, frames, digest.hexdigest(), owner=owner)
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            raise AsrError(409, "normalized_asset_invalid", "saved normalized audio could not be verified") from exc

    @classmethod
    def from_metadata(cls, path: Path, metadata: dict[str, Any], *,
                      spec: AudioSpec = DEFAULT_AUDIO_SPEC) -> NormalizedAudioAsset:
        if metadata.get("normalization_version") != NORMALIZATION_VERSION or metadata.get("spec") != asdict(spec):
            raise AsrError(409, "normalized_asset_version_mismatch", "saved normalization version or input specification differs")
        asset = cls.from_path(path, spec=spec, owner=str(metadata.get("owner", "workspace")))
        if asset.frame_count != metadata.get("frame_count") or asset.pcm_sha256 != metadata.get("pcm_sha256"):
            raise AsrError(409, "normalized_asset_invalid", "saved normalized audio content differs")
        return asset

    def validate(self, *, spec: AudioSpec | None = None) -> None:
        # Objects originate from bounded decoding or a full recovery hash.
        # Borrowing an unchanged, service-owned file need not scan hours of PCM.
        try:
            if (spec is None or spec == self.spec) and self._stat_identity() == self._file_identity:
                return
        except OSError:
            pass
        self.from_metadata(self.path, self.to_metadata(), spec=spec or self.spec)
