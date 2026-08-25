"""
generate_word_json.py のテスト

実際のネットワークアクセスは行わず、requests.get をモックして検証する。
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_word_json import ensure_vector_file, fetch_lyrics  # noqa: E402


class TestFetchLyrics:
    def _mock_response(self, songs):
        mock_response = MagicMock()
        mock_response.json.return_value = songs
        mock_response.raise_for_status.return_value = None
        return mock_response

    def test_returns_lyrics_of_eligible_songs(self):
        songs = [
            {"lyrics": "歌詞1", "is_joke": False, "is_questionable": False},
            {"lyrics": "歌詞2", "is_joke": False, "is_questionable": False},
        ]
        with patch("generate_word_json.requests.get", return_value=self._mock_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == ["歌詞1", "歌詞2"]

    def test_excludes_joke_songs(self):
        songs = [{"lyrics": "歌詞1", "is_joke": True, "is_questionable": False}]
        with patch("generate_word_json.requests.get", return_value=self._mock_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == []

    def test_excludes_questionable_songs(self):
        songs = [{"lyrics": "歌詞1", "is_joke": False, "is_questionable": True}]
        with patch("generate_word_json.requests.get", return_value=self._mock_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == []

    def test_excludes_songs_without_lyrics(self):
        songs = [{"lyrics": "", "is_joke": False, "is_questionable": False}]
        with patch("generate_word_json.requests.get", return_value=self._mock_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == []

    def test_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP error")
        with patch("generate_word_json.requests.get", return_value=mock_response):
            with pytest.raises(Exception):
                fetch_lyrics("https://example.com/api/song/")


class TestEnsureVectorFile:
    def _mock_stream_response(self, content, content_length=None):
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-Length": str(content_length or len(content))}
        mock_response.iter_content.return_value = [content]
        return mock_response

    def test_returns_existing_path_without_downloading(self, tmp_path):
        path = tmp_path / "cc.ja.300.vec.gz"
        path.write_bytes(b"dummy")

        with patch("generate_word_json.requests.get") as mock_get:
            result = ensure_vector_file(path=str(path), url="https://example.com/vec.gz")

        mock_get.assert_not_called()
        assert result == str(path)

    def test_downloads_when_missing(self, tmp_path):
        path = tmp_path / "cc.ja.300.vec.gz"
        mock_response = self._mock_stream_response(b"0123456789")

        with patch("generate_word_json.requests.get", return_value=mock_response) as mock_get:
            result = ensure_vector_file(path=str(path), url="https://example.com/vec.gz")

        mock_get.assert_called_once()
        assert os.path.exists(result)
        assert not os.path.exists(str(path) + ".part")
        assert path.read_bytes() == b"0123456789"

    def test_partial_file_removed_on_failure(self, tmp_path):
        path = tmp_path / "cc.ja.300.vec.gz"

        with patch("generate_word_json.requests.get", side_effect=Exception("network error")):
            with pytest.raises(Exception):
                ensure_vector_file(path=str(path), url="https://example.com/vec.gz")

        assert not os.path.exists(str(path) + ".part")
        assert not path.exists()
