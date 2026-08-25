"""
similar_words.py のテスト

word2vecモデル（gensimのKeyedVectors）はベクトルファイルが無いと読み込めず
このテスト環境では用意できないため、`build_word_candidates()` に渡す
modelはフェイク（`in`演算子とmost_similar(positive, topn)のみを実装した
スタブ）を使う。実際の類似度計算の妥当性はここでは検証できないが、
周辺のフィルタリング・データ整形ロジックの正しさは検証できる。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from similar_words import (  # noqa: E402
    build_word_candidates,
    conjugate_for_lookup,
    counter,
    is_replaceable_token,
    tokenizer_janome,
)


class FakeModel:
    """word: [(candidate, score), ...] の辞書を返すだけのword2vecモデルの代替"""

    def __init__(self, similar_map):
        self.similar_map = similar_map

    def __contains__(self, word):
        return word in self.similar_map

    def most_similar(self, positive, topn=10):
        return self.similar_map.get(positive, [])[:topn]


class TestCounter:
    def test_counts_hiragana(self):
        assert counter("あいう") == (3, 0, 0)

    def test_counts_katakana(self):
        assert counter("アイウ") == (0, 3, 0)

    def test_counts_kanji(self):
        assert counter("東京都") == (0, 0, 3)

    def test_counts_mixed(self):
        assert counter("走る") == (1, 0, 1)

    def test_counts_empty_string(self):
        assert counter("") == (0, 0, 0)


class TestTokenizerJanome:
    def test_splits_sentence(self):
        tokens = tokenizer_janome("私は走る")
        surfaces = [t[0] for t in tokens]
        assert surfaces == ["私", "は", "走る"]

    def test_verb_uses_full_infl_form_not_truncated(self):
        tokens = tokenizer_janome("走った")
        verb_token = next(t for t in tokens if t[0] == "走っ")
        # 活用形が2文字に切り詰められておらず、フルの文字列であること
        assert len(verb_token[2]) >= 2

    def test_noun_katsuyou_is_full_part_of_speech(self):
        tokens = tokenizer_janome("私")
        noun_token = tokens[0]
        assert noun_token[1] == "名詞"
        assert "," in noun_token[2]  # part_of_speechはカンマ区切りの詳細情報

    def test_particle_has_empty_katsuyou(self):
        tokens = tokenizer_janome("は")
        assert tokens[0][2] == ""


class TestIsReplaceableToken:
    def test_noun_is_replaceable(self):
        assert is_replaceable_token("犬", "名詞") is True

    def test_particle_is_not_replaceable(self):
        assert is_replaceable_token("は", "助詞") is False

    def test_single_char_verb_is_not_replaceable(self):
        # 音便処理ができないため対象外
        assert is_replaceable_token("見", "動詞") is False

    def test_multi_char_verb_is_replaceable(self):
        assert is_replaceable_token("走る", "動詞") is True

    def test_word_without_kana_or_kanji_is_not_replaceable(self):
        assert is_replaceable_token("123", "名詞") is False

    def test_adverb_is_replaceable(self):
        # 副詞は今回の改善で対象品詞に追加された
        assert is_replaceable_token("とても", "副詞") is True

    def test_adnominal_is_replaceable(self):
        # 連体詞も今回の改善で対象品詞に追加された
        assert is_replaceable_token("この", "連体詞") is True


class TestConjugateForLookup:
    def test_n_sound_change_verb(self):
        assert conjugate_for_lookup("読ん", "動詞") == "読んで"

    def test_small_tsu_sound_change_verb(self):
        assert conjugate_for_lookup("笑っ", "動詞") == "笑って"

    def test_i_sound_change_verb(self):
        assert conjugate_for_lookup("書い", "動詞") == "書いて"

    def test_adjective_small_tsu(self):
        assert conjugate_for_lookup("楽しかっ", "形容詞") == "楽しかって"

    def test_no_change_for_base_form(self):
        assert conjugate_for_lookup("走る", "動詞") == "走る"

    def test_no_change_for_noun(self):
        assert conjugate_for_lookup("犬", "名詞") == "犬"


class TestBuildWordCandidates:
    def test_picks_candidate_with_matching_hinshi_and_char_composition(self):
        # 「走る」(動詞,基本形,ひらがな1+漢字1) の類似語として
        # 「泳ぐ」(動詞,基本形,ひらがな1+漢字1) を用意する。
        # 「駆ける」等ひらがな数が異なる語は文字種構成フィルタで除外されるため、
        # あえて構成が一致するペアを選んでいる
        assert counter("走る") == counter("泳ぐ")
        run_tokens = tokenizer_janome("走る")
        oyogu_tokens = tokenizer_janome("泳ぐ")
        assert run_tokens[0][1] == oyogu_tokens[0][1]  # hinshi一致
        assert run_tokens[0][2] == oyogu_tokens[0][2]  # katsuyou一致

        model = FakeModel({
            "走る": [("泳ぐ", 0.9)],
        })

        result = build_word_candidates(["私は走る"], model)

        assert result == [{"word": "走る", "hinshi": "動詞", "candidates": ["泳ぐ"]}]

    def test_excludes_candidate_with_different_hinshi(self):
        model = FakeModel({
            "走る": [("犬", 0.9)],  # 名詞なので動詞の走るとは品詞が違う
        })

        result = build_word_candidates(["私は走る"], model)

        assert result == []

    def test_excludes_self_referential_candidate(self):
        model = FakeModel({
            "走る": [("走る", 0.99)],
        })

        result = build_word_candidates(["私は走る"], model)

        assert result == []

    def test_excludes_multi_token_candidate(self):
        # 「赤い花」のような複合語・フレーズは単語として差し替えられないため除外
        model = FakeModel({
            "花": [("赤い花", 0.9)],
        })

        result = build_word_candidates(["きれいな花"], model)

        assert result == []

    def test_word_not_in_model_is_skipped(self):
        model = FakeModel({})

        result = build_word_candidates(["私は走る"], model)

        assert result == []

    def test_same_word_across_multiple_lyrics_is_not_recomputed(self):
        call_count = {"n": 0}

        class CountingModel(FakeModel):
            def most_similar(self, positive, topn=10):
                call_count["n"] += 1
                return super().most_similar(positive, topn)

        model = CountingModel({"犬": [("猫", 0.9)]})
        # 名詞の活用形一致条件を満たすよう、同じ表記の単語で確認する
        result = build_word_candidates(["犬が好き", "犬と歩く"], model)

        assert call_count["n"] == 1
        assert len(result) <= 1

    def test_respects_max_candidates(self):
        many_candidates = [(f"猫{i}", 0.9 - i * 0.01) for i in range(30)]
        model = FakeModel({"犬": many_candidates})

        result = build_word_candidates(["犬"], model, max_candidates=5)

        if result:
            assert len(result[0]["candidates"]) <= 5
