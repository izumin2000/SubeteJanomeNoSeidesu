"""
Song APIから歌詞を取得し、word2vecモデルで模倣単語候補を計算して
word.json（subekashiの`manage.py word`コマンドが読み込む形式）を出力する。

実行にはword2vecモデルファイル（例: cc.ja.300.vec.gz）が別途必要。

使い方:
    python generate_word_json.py --vector-file cc.ja.300.vec.gz
"""
import argparse
import json

import gensim
import requests

from similar_words import build_word_candidates

DEFAULT_SONG_API = "https://subekashi.izmn.net/api/song/"


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
    parser.add_argument("--vector-file", default="cc.ja.300.vec.gz", help="word2vecモデルファイルのパス")
    parser.add_argument("--song-api", default=DEFAULT_SONG_API, help="Song APIのベースURL")
    parser.add_argument("--output", default="word.json", help="出力先ファイルパス")
    parser.add_argument("--max-candidates", type=int, default=20, help="単語ごとに保存する候補数の上限")
    args = parser.parse_args()

    print(f"歌詞を取得中... ({args.song_api})")
    lyrics_list = fetch_lyrics(args.song_api)
    print(f"{len(lyrics_list)}件の歌詞を取得しました")

    print(f"word2vecモデルを読み込み中... ({args.vector_file})")
    model = gensim.models.KeyedVectors.load_word2vec_format(args.vector_file)

    print("模倣単語候補を計算中...")
    word_candidates = build_word_candidates(lyrics_list, model, max_candidates=args.max_candidates)
    print(f"{len(word_candidates)}語分の候補を生成しました")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(word_candidates, f, ensure_ascii=False, indent=2)
    print(f"{args.output} に出力しました")


if __name__ == "__main__":
    main()
