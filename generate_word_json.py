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

import gensim
import requests

from similar_words import build_word_candidates

DEFAULT_SONG_API = "https://subekashi.izmn.net/api/song/"

VECTOR_FILE = "cc.ja.300.vec.gz"
VECTOR_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"


def ensure_vector_file(path=VECTOR_FILE, url=VECTOR_URL):
    """
    word2vecベクトルファイルがローカルに無ければダウンロードする。
    ダウンロードが中断された場合に不完全なファイルを正常なものとして
    扱わないよう、一時ファイルに書き込んでから完了後にリネームする。
    """
    if os.path.exists(path):
        return path

    print(f"{path} が見つからないため、ダウンロードします: {url}")
    tmp_path = path + ".part"
    try:
        with requests.get(url, stream=True, timeout=60) as res:
            res.raise_for_status()
            total = int(res.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded * 100 // total
                        print(
                            f"\rダウンロード中... {percent}% "
                            f"({downloaded // (1024 * 1024)}MB / {total // (1024 * 1024)}MB)",
                            end="",
                        )
            print()
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    os.replace(tmp_path, path)
    return path


def fetch_lyrics(song_api_url=DEFAULT_SONG_API):
    """
    Song APIから歌詞テキストの一覧を取得する。
    ネタ動画（is_joke）・界隈曲か疑わしい曲（is_questionable）は除外する。
    """
    response = requests.get(f"{song_api_url}?format=json")
    response.raise_for_status()
    songs = response.json()

    return [
        song["lyrics"]
        for song in songs
        if song.get("lyrics") and not song.get("is_joke") and not song.get("is_questionable")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song-api", default=DEFAULT_SONG_API, help="Song APIのベースURL")
    parser.add_argument("--output", default="word.json", help="出力先ファイルパス")
    parser.add_argument("--max-candidates", type=int, default=20, help="単語ごとに保存する候補数の上限")
    args = parser.parse_args()

    print(f"歌詞を取得中... ({args.song_api})")
    lyrics_list = fetch_lyrics(args.song_api)
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
