"""
歌詞のトークナイズと、word2vecモデルを使った模倣単語候補の抽出ロジック。

`build_word_candidates()` に歌詞のリストと読み込み済みのword2vecモデル
（gensimのKeyedVectors互換：`in`演算子と`most_similar(positive=..., topn=...)`
が使えるもの）を渡すと、subekashi側の `manage.py word` が読み込む
word.json形式のデータを構築できる。
"""
import re
import string
import unicodedata

from janome.tokenizer import Tokenizer

# subekashi側の REPLACEABLE_HINSHIS（subekashi/lib/lyric_tokenizer.py）に、
# 副詞・連体詞を加えて置き換え対象の品詞を増やした。
# ここで増やした品詞をsubekashi側でも実際に使うには、
# subekashi側のREPLACEABLE_HINSHISも合わせて更新する必要がある。
REPLACEABLE_HINSHIS = ["名詞", "動詞", "形容詞", "副詞", "連体詞"]

# 五段動詞の連用タ接続（例:「読ん」「書い」）に対する、正しい撥音便/
# 促音便/イ音便の接尾辞。infl_type（活用の種類）ごとに異なるため、
# 語幹の末尾の文字だけでは正しく判定できない
# （例:「泳い」「書い」はどちらも「い」で終わるが、正しい語彙表記は
# 「泳いで」「書いて」と濁る/濁らないが異なる）。
_VERB_ONBIN_SUFFIX = {
    "五段・ガ行": "で",
    "五段・カ行イ音便": "て",
    "五段・マ行": "で",
    "五段・ナ行": "で",
    "五段・バ行": "で",
    "五段・ワ行促音便": "て",
    "五段・タ行": "て",
    "五段・ラ行": "て",
    # 五段・サ行（例:「話し」）・一段（例:「食べ」）は音便が発生せず、
    # そのまま「て」「た」等が直接続くためテーブルに含めない
}

_tokenizer = Tokenizer()


def counter(word):
    """ひらがな・カタカナ・漢字それぞれの文字数を返す"""
    word = str(word)
    hiragana = sum(1 for c in word if "ぁ" <= c <= "ゟ")
    katakana = sum(1 for c in word if "ァ" <= c <= "ヿ")
    kanji = sum(1 for c in word if "一" <= c <= "鿼")
    return hiragana, katakana, kanji


def tokenizer_janome(text):
    """
    テキストを単語ごとに分割し、(表記, 品詞, 活用形/品詞細分類, 活用の種類)
    のタプルのリストを返す。

    動詞・形容詞は活用形(infl_form)をそのまま、名詞は品詞細分類
    (part_of_speech)をそのまま3つ目の要素とする。先頭2文字だけを見る
    実装だと、例えば「連用形」と「連用タ接続」のように異なる活用形を
    同一視してしまうため、切り詰めは行わない。
    4つ目の要素（活用の種類、infl_type）は動詞の音便判定にのみ使う。
    """
    toklist = []
    for tok in _tokenizer.tokenize(text, wakati=False):
        hinshi = tok.part_of_speech.split(',')[0]
        if hinshi in ("動詞", "形容詞"):
            katsuyou = tok.infl_form
            infl_type = tok.infl_type
        elif hinshi == "名詞":
            katsuyou = tok.part_of_speech
            infl_type = ""
        else:
            katsuyou = ""
            infl_type = ""
        toklist.append((tok.surface, hinshi, katsuyou, infl_type))
    return toklist


def format_text(text):
    text = unicodedata.normalize("NFKC", text)
    table = str.maketrans("", "", string.punctuation + "「」、。・")
    return text.translate(table)


def preprocessing(text):
    """前処理をする関数"""
    text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))  # 全角→半角
    text = text.lower()  # 大文字→小文字

    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub(' ', '', text)
    text = re.sub('　', '', text)

    text = re.sub(r'[0-9 ０-９]', '', text)  # 数字の除去
    text = re.sub(r'[!-/:-@[-`{-~]', '', text)  # 半角記号の除去
    text = re.sub(r'[！-／：-＠［-｀｛-～、-〜”’・]', '', text)  # 全角記号の除去
    text = format_text(text)

    return text


def tokenizer_with_preprocessing(text):
    return tokenizer_janome(preprocessing(text))


def conjugate_for_lookup(word, hinshi, katsuyou="", infl_type=""):
    """
    word2vecの語彙は言い切りに近い表記（例:「読んで」）で登録されている
    ことが多いため、連用タ接続（katsuyou=="連用タ接続"）の語幹に対して、
    活用の種類(infl_type)に応じた正しい撥音便・促音便・イ音便の接尾辞を
    補ってから語彙検索する。
    形容詞の連用タ接続は常に「て」（例:「楽しかっ」→「楽しかって」）。
    サ行五段・一段活用の動詞（連用タ接続ではなく連用形になる、
    例:「話し」「食べ」）や基本形は音便が発生しないためそのまま検索する。
    戻り値は検索専用の表記であり、候補として保存する表記（引数のword）
    自体は書き換えない。
    """
    if katsuyou != "連用タ接続":
        return word
    if hinshi == "動詞":
        suffix = _VERB_ONBIN_SUFFIX.get(infl_type)
        return word + suffix if suffix else word
    if hinshi == "形容詞":
        return word + "て"
    return word


def is_replaceable_token(word, hinshi):
    """置き換え候補として扱ってよい単語かどうかを判定する"""
    if hinshi not in REPLACEABLE_HINSHIS:
        return False
    if not sum(counter(word)):
        return False
    if hinshi == "動詞" and len(word) <= 1:
        # 1文字の動詞は音便処理（conjugate_for_lookup）の対象にできないため除外
        return False
    return True


def build_word_candidates(lyrics_list, model, max_candidates=20, oversample_factor=3):
    """
    歌詞のリストとword2vecモデルから、模倣単語候補データを構築する。

    同じ (word, hinshi, katsuyou) の組み合わせは歌詞をまたいで重複計算
    しない（katsuyouまで含めるのは、同じ表記・品詞大分類でも活用形/
    品詞細分類が異なる出現に対して、最初の出現用に絞り込んだ候補を
    誤って使い回さないようにするため）。
    候補は以下の条件をすべて満たすもののみ採用する：
    - 品詞（大分類）が元の単語と一致する
    - 活用形/品詞細分類が元の単語と一致する（差し替えても文法的に破綻しにくくするため）
    - ひらがな・カタカナ・漢字の構成が元の単語と一致する（見た目のバランスを保つため）
    - 単一のjanomeトークンとして解釈できる（複合語・フレーズ候補を除外するため）

    戻り値: [{"word": str, "hinshi": str, "candidates": [str, ...]}, ...]
    候補が1件も残らなかった単語は結果に含まれない。
    """
    results = {}
    computed = set()

    for lyrics in lyrics_list:
        for word, hinshi, katsuyou, infl_type in tokenizer_with_preprocessing(lyrics):
            memo_key = (word, hinshi, katsuyou)
            if memo_key in computed or not is_replaceable_token(word, hinshi):
                continue
            computed.add(memo_key)

            lookup_word = conjugate_for_lookup(word, hinshi, katsuyou, infl_type)
            if lookup_word not in model:
                continue

            topn = max(max_candidates * oversample_factor, max_candidates)
            sims = model.most_similar(positive=lookup_word, topn=topn)

            candidates = []
            for sim_word, _score in sims:
                sim_tokens = tokenizer_janome(sim_word)
                if len(sim_tokens) != 1:
                    continue  # 複合語・フレーズはそのまま単語として差し替えられないため除外
                sim_surface, sim_hinshi, sim_katsuyou, _sim_infl_type = sim_tokens[0]
                if sim_surface == word:
                    continue
                if not is_replaceable_token(sim_surface, sim_hinshi):
                    continue
                if sim_hinshi != hinshi or sim_katsuyou != katsuyou:
                    continue
                if counter(word) != counter(sim_surface):
                    continue
                candidates.append(sim_surface)
                if len(candidates) >= max_candidates:
                    break

            if candidates:
                results[(word, hinshi)] = candidates

    return [
        {"word": word, "hinshi": hinshi, "candidates": candidates}
        for (word, hinshi), candidates in results.items()
    ]
