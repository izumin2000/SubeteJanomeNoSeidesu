"""
generate_word_json.py のテスト

実際のネットワークアクセスは行わず、requests.get をモックして検証する。
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_word_json import ensure_vector_file, fetch_lyrics  # noqa: E402


class TestFetchLyrics:
    """
    Song APIのレスポンスは {"result": [...], "page": int, "max_page": int, ...}
    形式のページネーションされたオブジェクトである（実際のAPIレスポンスで確認済み）。
    """

    def _mock_page_response(self, songs, page=1, max_page=1):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": songs,
            "page": page,
            "max_page": max_page,
            "count": len(songs),
            "size": len(songs),
        }
        mock_response.raise_for_status.return_value = None
        return mock_response

    def test_returns_lyrics_of_eligible_songs(self):
        songs = [
            {"lyrics": "歌詞1", "is_joke": False, "is_questionable": False},
            {"lyrics": "歌詞2", "is_joke": False, "is_questionable": False},
        ]
        with patch("generate_word_json.requests.get", return_value=self._mock_page_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == ["歌詞1", "歌詞2"]

    def test_excludes_joke_songs(self):
        songs = [{"lyrics": "歌詞1", "is_joke": True, "is_questionable": False}]
        with patch("generate_word_json.requests.get", return_value=self._mock_page_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == []

    def test_excludes_questionable_songs(self):
        songs = [{"lyrics": "歌詞1", "is_joke": False, "is_questionable": True}]
        with patch("generate_word_json.requests.get", return_value=self._mock_page_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == []

    def test_excludes_songs_without_lyrics(self):
        songs = [{"lyrics": "", "is_joke": False, "is_questionable": False}]
        with patch("generate_word_json.requests.get", return_value=self._mock_page_response(songs)):
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == []

    def test_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP error")
        with patch("generate_word_json.requests.get", return_value=mock_response):
            with pytest.raises(Exception):
                fetch_lyrics("https://example.com/api/song/")

    def test_traverses_all_pages(self):
        page1 = self._mock_page_response(
            [{"lyrics": "歌詞1", "is_joke": False, "is_questionable": False}], page=1, max_page=3
        )
        page2 = self._mock_page_response(
            [{"lyrics": "歌詞2", "is_joke": False, "is_questionable": False}], page=2, max_page=3
        )
        page3 = self._mock_page_response(
            [{"lyrics": "歌詞3", "is_joke": False, "is_questionable": False}], page=3, max_page=3
        )

        with patch("generate_word_json.requests.get", side_effect=[page1, page2, page3]) as mock_get:
            result = fetch_lyrics("https://example.com/api/song/")

        assert result == ["歌詞1", "歌詞2", "歌詞3"]
        assert mock_get.call_count == 3

    def test_passes_page_size_param(self):
        with patch(
            "generate_word_json.requests.get", return_value=self._mock_page_response([])
        ) as mock_get:
            fetch_lyrics("https://example.com/api/song/", page_size=123)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["size"] == 123


class TestEnsureVectorFile:
    def _mock_stream_response(self, content, status_code=200, content_length=None):
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = status_code
        mock_response.headers = {"Content-Length": str(content_length if content_length is not None else len(content))}
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

    def test_partial_file_removed_after_exhausting_retries(self, tmp_path):
        path = tmp_path / "cc.ja.300.vec.gz"

        with patch(
            "generate_word_json.requests.get",
            side_effect=requests.exceptions.ConnectionError("network error"),
        ):
            with patch("generate_word_json.time.sleep"):
                with pytest.raises(requests.exceptions.RequestException):
                    ensure_vector_file(path=str(path), url="https://example.com/vec.gz", max_retries=1)

        assert not os.path.exists(str(path) + ".part")
        assert not path.exists()

    def test_retries_after_transient_failure_then_succeeds(self, tmp_path):
        path = tmp_path / "cc.ja.300.vec.gz"
        success_response = self._mock_stream_response(b"0123456789")

        with patch(
            "generate_word_json.requests.get",
            side_effect=[requests.exceptions.ConnectionError("transient"), success_response],
        ) as mock_get:
            with patch("generate_word_json.time.sleep") as mock_sleep:
                result = ensure_vector_file(path=str(path), url="https://example.com/vec.gz", max_retries=3)

        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()
        assert path.read_bytes() == b"0123456789"
        assert result == str(path)

    def test_resumes_from_partial_download_via_range_request(self, tmp_path):
        path = tmp_path / "cc.ja.300.vec.gz"
        tmp_path_file = tmp_path / "cc.ja.300.vec.gz.part"
        tmp_path_file.write_bytes(b"01234")  # 前回の続きから取得する想定

        resume_response = self._mock_stream_response(b"56789", status_code=206)

        with patch("generate_word_json.requests.get", return_value=resume_response) as mock_get:
            result = ensure_vector_file(path=str(path), url="https://example.com/vec.gz")

        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == {"Range": "bytes=5-"}
        assert path.read_bytes() == b"0123456789"
        assert result == str(path)

    def test_restarts_from_scratch_when_server_ignores_range(self, tmp_path):
        # サーバーがRangeに対応しておらず200（部分取得でなく全体）を返した場合、
        # 中途半端な.partファイルの内容を引き継がず最初からやり直す
        path = tmp_path / "cc.ja.300.vec.gz"
        tmp_path_file = tmp_path / "cc.ja.300.vec.gz.part"
        tmp_path_file.write_bytes(b"stale-partial-data")

        full_response = self._mock_stream_response(b"0123456789", status_code=200)

        with patch("generate_word_json.requests.get", return_value=full_response):
            result = ensure_vector_file(path=str(path), url="https://example.com/vec.gz")

        assert path.read_bytes() == b"0123456789"
        assert result == str(path)
