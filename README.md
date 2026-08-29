# 全て蛇の目の所為です。

「全て歌詞の所為です。」（subekashi）に登録された曲の歌詞をjanomeで形態素解析し、word2vecで類似度の高い模倣単語候補を抽出するリポジトリ。

出力した `word.json` は、subekashiリポジトリの `python manage.py word` コマンドで `Word` モデルに取り込む。これがsubekashi側の「生成歌詞の単語クリック入れ替え機能」の候補データになる。

## 構成

- `similar_words.py`: 歌詞のトークナイズと模倣単語候補の抽出ロジック（word2vecモデル非依存の部分、テスト対象）
- `generate_word_json.py`: Song APIから歌詞を取得し、word2vecモデルを使って `word.json` を出力するスクリプト
- `tests/test_similar_words.py`: `similar_words.py` の単体テスト（word2vecモデルはフェイクを使用し、ロジック部分のみ検証）

## セットアップ（ローカル）

```
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

`similar_words.py` 自体はword2vecモデルを直接読み込まないため、word2vecベクトルファイル無しでもロジックのテストは実行できる。

```
pytest tests/
```

## word.jsonの生成

```
python generate_word_json.py
```

word2vecモデルはfastTextの日本語学習済みベクトル（`cc.ja.300.vec.gz`、圧縮状態で約1.2GB）に固定している。カレントディレクトリに無ければ自動的にダウンロードする（初回のみ時間がかかる）。既に手元にある場合は、カレントディレクトリに `cc.ja.300.vec.gz` という名前で置いておけばダウンロードをスキップする。

生成された `word.json` を、subekashiリポジトリの `subekashi/constants/dynamic/word.json` に配置して `python manage.py word` を実行すると、`Word` モデルに取り込まれる。

## word.jsonの形式

```json
[
  {"word": "走る", "hinshi": "動詞", "katsuyou": "基本形", "candidates": ["泳ぐ", "駆け出す"]},
  {"word": "犬", "hinshi": "名詞", "katsuyou": "名詞,一般,*,*", "candidates": ["猫", "狼"]}
]
```

`katsuyou`は、動詞・形容詞は活用形（`infl_form`）、名詞は品詞細分類（`part_of_speech`のフル文字列）、それ以外は空文字列。subekashi側で候補を絞り込む際、`word`ではなく`hinshi`・`katsuyou`の組み合わせで一致判定するために使う（同じ`word`でなくても、活用形が一致していれば置き換えても文法的に破綻しにくいため）。

## 候補選定のルール

- 対象品詞: 名詞・動詞・形容詞・副詞・連体詞
- 元の単語と品詞（大分類）・活用形/品詞細分類が一致する候補のみ採用（差し替え後も文法的に破綻しにくくするため）
- 元の単語とひらがな・カタカナ・漢字の構成数が一致する候補のみ採用（見た目のバランスを保つため）
- 単一のjanomeトークンとして解釈できる候補のみ採用（複合語・フレーズを除外するため）

上記の条件をすべて満たす候補が無い単語は `word.json` に含まれない（= subekashi側でその単語はクリックできない）。品詞・文字種条件を厳しめにしているため、実際にどの程度候補が集まるかは実データで要確認。
