"""Real-byte regressions for legacy and mislabeled attachment extraction."""
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from extractors import begin_degradation_capture, collect_degradations, extract_text


class InspectingProvider:
    def transcribe_audio(self, path):
        path = Path(path)
        assert path.suffix == '.wav'
        info = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_name',
            '-of', 'csv=p=0', str(path),
        ], text=True)
        assert 'pcm_s16le' in info
        return 'Spoken maintenance request'

    def analyze_video(self, path):
        assert Path(path).suffix == '.mp4'
        return 'Visible property walkthrough'

    def describe(self, path):
        with Image.open(path) as image:
            assert image.format in {'PNG', 'JPEG'}
        return 'Visible maintenance photo'


@pytest.mark.parametrize('suffix', ['3gp', 'bin', 'wav'])
def test_video_bytes_route_as_video_even_with_audio_suffix(tmp_path, suffix):
    path = tmp_path / f'clip.{suffix}'
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=32x32:d=0.1',
        '-c:v', 'mpeg4', '-f', '3gp', str(path),
    ], check=True)
    original = path.read_bytes()
    result = extract_text(path, suffix, media_provider=InspectingProvider())
    assert result.full_text == 'Visible property walkthrough'
    assert result.frontmatter['media_type'] == 'video'
    assert path.read_bytes() == original


@pytest.mark.parametrize('suffix', ['wav', 'bin', 'amr'])
def test_wav_bytes_normalized_even_with_wrong_suffix(tmp_path, suffix):
    path = tmp_path / f'voice.{suffix}'
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=duration=0.1',
        '-c:a', 'pcm_mulaw', '-f', 'wav', str(path),
    ], check=True)
    original = path.read_bytes()
    result = extract_text(path, suffix, media_provider=InspectingProvider())
    assert result.full_text == 'Spoken maintenance request'
    assert result.frontmatter['media_type'] == 'audio'
    assert path.read_bytes() == original


@pytest.mark.parametrize('suffix', ['heic', 'bin', 'jpg'])
def test_heic_bytes_converted_before_vision(tmp_path, suffix):
    from pillow_heif import register_heif_opener
    register_heif_opener()
    path = tmp_path / f'photo.{suffix}'
    Image.new('RGB', (32, 32), 'blue').save(path, format='HEIF')
    result = extract_text(path, suffix, ocr_provider=InspectingProvider())
    assert 'Visible maintenance photo' in result.full_text
    assert result.primary_content


@pytest.mark.parametrize('suffix', ['tiff', 'tif', 'bin', 'jpg'])
def test_tiff_bytes_converted_before_vision(tmp_path, suffix):
    path = tmp_path / f'scan.{suffix}'
    Image.new('RGB', (32, 32), 'blue').save(path, format='TIFF')
    original = path.read_bytes()
    result = extract_text(path, suffix, ocr_provider=InspectingProvider())
    assert 'Visible maintenance photo' in result.full_text
    assert result.primary_content
    assert result.frontmatter['media_type'] == 'img'
    assert path.read_bytes() == original


def test_multipage_tiff_extracts_every_page_and_removes_temporaries(tmp_path):
    path = tmp_path / 'scan.bin'
    Image.new('RGB', (32, 32), 'red').save(
        path, format='TIFF', save_all=True,
        append_images=[Image.new('RGB', (32, 32), 'blue')],
    )
    original = path.read_bytes()
    paths = []

    class PageProvider:
        def describe(self, prepared):
            paths.append(Path(prepared))
            with Image.open(prepared) as page:
                assert page.format == 'PNG'
                return 'Red page' if page.getpixel((0, 0)) == (255, 0, 0) else 'Blue page'

    result = extract_text(path, 'bin', ocr_provider=PageProvider())
    assert 'Page 1:\nRed page' in result.full_text
    assert 'Page 2:\nBlue page' in result.full_text
    assert result.primary_content
    assert len(paths) == 2 and all(not page.exists() for page in paths)
    assert path.read_bytes() == original


def test_unknown_bin_is_missing_with_diagnostic(tmp_path):
    path = tmp_path / 'unknown.bin'
    path.write_bytes(b'not recognized media')
    begin_degradation_capture()
    result = extract_text(path, 'bin', media_provider=InspectingProvider())
    assert not result.full_text
    assert not result.primary_content
    assert 'unrecognized_media_format' in [item.reason for item in collect_degradations()]


def test_multipage_tiff_confirmed_blank_has_no_primary_content(tmp_path):
    path = tmp_path / 'blank.bin'
    Image.new('RGB', (32, 32), 'white').save(
        path, format='TIFF', save_all=True,
        append_images=[Image.new('RGB', (32, 32), 'white')],
    )

    class BlankProvider:
        def describe(self, prepared):
            return ''

    begin_degradation_capture()
    result = extract_text(path, 'bin', ocr_provider=BlankProvider())
    assert not result.primary_content
    assert collect_degradations() == []


def test_real_amr_normalizes_before_transcription(tmp_path):
    path = tmp_path / 'voice.amr'
    # One narrowband AMR mode-7 frame, enough to exercise actual decoder.
    path.write_bytes(b'#!AMR\n' + bytes([0x3c]) + bytes(31))
    result = extract_text(path, 'amr', media_provider=InspectingProvider())
    assert result.full_text == 'Spoken maintenance request'


def test_temporary_conversion_removed_after_provider_failure(tmp_path):
    from core.media_formats import provider_media_path
    path = tmp_path / 'voice.amr'
    path.write_bytes(b'#!AMR\n' + bytes([0x3c]) + bytes(31))
    with pytest.raises(RuntimeError):
        with provider_media_path(path, 'amr', 'audio') as prepared:
            assert prepared.exists()
            raise RuntimeError('provider failed')
    assert not prepared.exists()
    assert path.exists()


def test_conversion_failure_is_missing_and_degraded(tmp_path):
    path = tmp_path / 'broken.amr'
    path.write_bytes(b'#!AMR\ninvalid')
    begin_degradation_capture()
    result = extract_text(path, 'amr', media_provider=InspectingProvider())
    assert not result.primary_content
    assert not result.full_text
    assert [item.reason for item in collect_degradations()] == ['audio_extract_failed']
