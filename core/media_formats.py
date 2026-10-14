"""Byte-based attachment routing and temporary provider-compatible conversion.

Original files, names, sidecars and registry identifiers are never changed.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterator

IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'heic', 'heif', 'tif', 'tiff'}
AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'aiff', 'amr', 'awb', 'mka'}
VIDEO_EXTENSIONS = {'mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v', '3gp', '3g2', 'ogv'}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | {'bin'}


class MediaConversionError(ValueError):
    """Local decoder/probe failed; capture this in the degraded-doc ledger."""


def _probe(path: Path) -> dict:
    try:
        completed = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type:format=format_name',
            '-of', 'json', str(path),
        ], capture_output=True, text=True, timeout=30, check=True)
        return json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise MediaConversionError('media probe failed') from exc


def detect_media_format(file_path: str | Path, extension: str) -> str:
    """Prefer recognized signatures over labels, probing ambiguous containers.

    Unknown .bin content stays unknown. In particular JSON error bodies and
    calendar attachments must never be sent to a media model.
    """
    ext = extension.lower().lstrip('.')
    if ext not in MEDIA_EXTENSIONS:
        return ext
    path = Path(file_path)
    try:
        with path.open('rb') as stream:
            header = stream.read(256)
    except OSError:
        return ext
    if header.startswith(b'%PDF-'):
        return 'pdf'
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if header.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    if header.startswith((b'GIF87a', b'GIF89a')):
        return 'gif'
    if header.startswith((b'II\x2a\x00', b'MM\x00\x2a', b'II\x2b\x00', b'MM\x00\x2b')):
        return 'tiff'
    if header.startswith(b'RIFF'):
        kind = header[8:12]
        if kind == b'WAVE':
            return 'wav'
        if kind == b'WEBP':
            return 'webp'
        if kind == b'AVI ':
            return 'avi'
    if header.startswith(b'FORM') and header[8:12] in {b'AIFF', b'AIFC'}:
        return 'aiff'
    if header.startswith(b'#!AMR'):
        return 'amr'
    if header.startswith(b'fLaC'):
        return 'flac'
    if header.startswith(b'ID3') or (len(header) > 1 and header[0] == 255 and header[1] & 0xE0 == 0xE0):
        return 'aac' if len(header) > 1 and header[1] & 0xF6 == 0xF0 else 'mp3'
    if header[4:8] == b'ftyp':
        brands = {header[i:i+4] for i in [8, *range(16, min(len(header), int.from_bytes(header[:4], 'big')), 4)]}
        if brands & {b'heic', b'heix', b'hevc', b'hevx', b'mif1', b'msf1'}:
            return 'heic'
        try:
            info = _probe(path)
        except MediaConversionError:
            if ext in {'mp4', 'mov', 'm4v'} and not any(b.startswith((b'3gp', b'3g2')) for b in brands):
                return ext
            raise
        has_video = any(s.get('codec_type') == 'video' for s in info.get('streams', []))
        if not has_video and any(s.get('codec_type') == 'audio' for s in info.get('streams', [])):
            return 'amr' if any(b.startswith((b'3gp', b'3g2')) for b in brands) else 'm4a'
        if not has_video:
            raise MediaConversionError('container has no audio/video stream')
        return '3gp' if any(b.startswith((b'3gp', b'3g2')) for b in brands) else 'mov' if b'qt  ' in brands else 'mp4'
    # Ogg and Matroska can contain either audio or video. Probe actual streams.
    if header.startswith((b'OggS', b'\x1aE\xdf\xa3')):
        info = _probe(path)
        video = any(s.get('codec_type') == 'video' for s in info.get('streams', []))
        if header.startswith(b'OggS'):
            return 'ogg' if not video else 'ogv'
        return 'webm' if video else 'mka'
    return ext


@contextmanager
def provider_media_path(file_path: str | Path, detected_format: str, media_type: str) -> Iterator[Path]:
    """Normalize legacy codecs; temporary paths live through the provider call."""
    path = Path(file_path)
    heif = detected_format in {'heic', 'heif'}
    audio = media_type == 'audio' and detected_format in {'wav', 'amr', 'awb', '3gp', '3g2', 'mka'}
    video = media_type == 'video' and detected_format in {'3gp', '3g2', 'ogv'}
    if not (heif or audio or video) and path.suffix.lower().lstrip('.') == detected_format:
        yield path
        return
    with TemporaryDirectory(prefix='organizer-media-') as directory:
        suffix = 'png' if heif else 'wav' if audio else 'mp4' if video else detected_format
        output = Path(directory) / f'{path.stem}.{suffix}'
        try:
            if heif:
                from pillow_heif import register_heif_opener
                from PIL import Image
                register_heif_opener()
                with Image.open(path) as image:
                    image.convert('RGB').save(output, format='PNG')
            elif audio or video:
                args = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(path)]
                if audio:
                    args += ['-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le']
                else:
                    args += ['-map', '0:v:0', '-map', '0:a:0?', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart']
                subprocess.run([*args, str(output)], capture_output=True, timeout=300, check=True)
            else:
                # Hard links avoid copying potentially large mislabeled recordings.
                try:
                    output.hardlink_to(path)
                except OSError:
                    shutil.copyfile(path, output)
        except (ImportError, OSError, subprocess.SubprocessError, ValueError) as exc:
            raise MediaConversionError('media conversion failed') from exc
        yield output


@contextmanager
def provider_image_paths(file_path: str | Path, detected_format: str) -> Iterator[list[Path]]:
    """Render every TIFF page separately; other images use existing conversion."""
    if detected_format not in {'tif', 'tiff'}:
        with provider_media_path(file_path, detected_format, 'img') as prepared:
            yield [prepared]
        return
    with TemporaryDirectory(prefix='organizer-media-') as directory:
        try:
            from PIL import Image
            pages = []
            with Image.open(file_path) as image:
                for index in range(image.n_frames):
                    image.seek(index)
                    output = Path(directory) / f'page-{index + 1}.png'
                    image.convert('RGB').save(output, format='PNG')
                    pages.append(output)
        except (ImportError, OSError, ValueError) as exc:
            raise MediaConversionError('image conversion failed') from exc
        yield pages
