"""
Song APIから歌詞を取得し、word2vecモデルで模倣単語候補を計算して
word.json（subekashiの`manage.py word`コマンドが読み込む形式）を出力する。

word2vecモデルはfastTextの日本語学習済みベクトル（cc.ja.300.vec.gz、
圧縮状態で約1.2GB）に固定している。カレントディレクトリに無い場合は
自動的にダウンロードする。

使い方:
    python generate_word_json.py
"""
import argparse
import json
import os
import time

import gensim
import requests

from similar_words import build_word_candidates

DEFAULT_SONG_API = "https://lyrics.imicomweb.com/api/song/"
DEFAULT_PAGE_SIZE = 500

VECTOR_FILE = "cc.ja.300.vec.gz"
VECTOR_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"


def ensure_vector_file(path=VECTOR_FILE, url=VECTOR_URL, max_retries=5):
    """
    word2vecベクトルファイル（約1.2GB）がローカルに無ければダウンロードする。

    不安定なネットワークでも完走できるよう、通信エラー発生時は
    Rangeリクエストでダウンロード済みの続きから再試行する
    （サーバーがRangeに対応していない場合は最初からやり直す）。
    ダウンロードが完了するまでは一時ファイル（.part）に書き込み、
    完了後にリネームすることで、不完全なファイルが正常なものとして
    扱われないようにする。
    """
    if os.path.exists(path):
        return path

    tmp_path = path + ".part"
    for attempt in range(1, max_retries + 1):
        try:
            downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

            with requests.get(url, stream=True, timeout=30, headers=headers) as res:
                if downloaded and res.status_code == 206:
                    mode = "ab"
                else:
                    res.raise_for_status()
                    downloaded = 0
                    mode = "wb"

                total = downloaded + int(res.headers.get("Content-Length", 0))
                with open(tmp_path, mode) as f:
                    for chunk in res.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 // total
                            print(
                                f"\rダウンロード中(試行{attempt}/{max_retries})... {percent}% "
                                f"({downloaded // (1024 * 1024)}MB / {total // (1024 * 1024)}MB)",
                                end="",
                            )
            print()
            os.replace(tmp_path, path)
            return path
        except (requests.exceptions.RequestException, OSError) as e:
            # OSErrorも捕捉するのは、ダウンロード中のファイルロック等
            # （例: Windows上のウイルス対策ソフトによる一時的なロック）で
            # os.path.getsize/open がエラーになるケースも再試行対象に
            # 含めるため
            print(f"\nダウンロードに失敗しました（試行{attempt}/{max_retries}）: {e}")
            if attempt == max_retries:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            time.sleep(min(2 ** attempt, 30))


def _fetch_song_page(song_api_url, page_size, page, max_retries=3):
    """1ページ分をリトライ付きで取得する（一時的な5xx等に備えるため）"""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                f"{song_api_url}?format=json",
                params={"size": page_size, "page": page},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueErrorはresponse.json()がJSONとしてパースできない場合
            # （メンテナンスページ等が返ってきたケース）を捕捉する
            print(f"\nページ{page}の取得に失敗しました（試行{attempt}/{max_retries}）: {e}")
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def fetch_lyrics(song_api_url=DEFAULT_SONG_API, page_size=DEFAULT_PAGE_SIZE):
    """
    Song APIから歌詞テキストの一覧を取得する。

    Song APIのレスポンスは {"result": [...], "page": int, "max_page": int, ...}
    形式のページネーションされたオブジェクトであるため、max_pageまで
    ページを辿って全件取得する。
    ネタ動画（is_joke）・界隈曲か疑わしい曲（is_questionable）は除外する。
    """
    lyrics_list = []
    page = 1
    while True:
        data = _fetch_song_page(song_api_url, page_size, page)

        lyrics_list.extend(
            song["lyrics"]
            for song in data["result"]
            if song.get("lyrics") and not song.get("is_joke") and not song.get("is_questionable")
        )

        if page >= data["max_page"]:
            break
        page += 1

    return lyrics_list


def _positive_int(value):
    int_value = int(value)
    if int_value < 1:
        raise argparse.ArgumentTypeError(f"1以上の整数を指定してください: {value}")
    return int_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song-api", default=DEFAULT_SONG_API, help="Song APIのベースURL")
    parser.add_argument("--page-size", type=_positive_int, default=DEFAULT_PAGE_SIZE, help="Song API取得時の1ページあたり件数")
    parser.add_argument("--output", default="word.json", help="出力先ファイルパス")
    parser.add_argument("--max-candidates", type=_positive_int, default=20, help="単語ごとに保存する候補数の上限")
    args = parser.parse_args()

    print(f"歌詞を取得中... ({args.song_api})")
    lyrics_list = fetch_lyrics(args.song_api, page_size=args.page_size)
    print(f"{len(lyrics_list)}件の歌詞を取得しました")

    vector_path = ensure_vector_file()
    print(f"word2vecモデルを読み込み中... ({vector_path})")
    model = gensim.models.KeyedVectors.load_word2vec_format(vector_path)

    print("模倣単語候補を計算中...")
    word_candidates = build_word_candidates(lyrics_list, model, max_candidates=args.max_candidates)
    print(f"{len(word_candidates)}語分の候補を生成しました")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(word_candidates, f, ensure_ascii=False, indent=2)
    print(f"{args.output} に出力しました")


if __name__ == "__main__":
    main()
