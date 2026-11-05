from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import kael

DATA = Path(__file__).resolve().parents[2] / "data" / "gpt2"
VOCABULARY = DATA / "vocab.json"
MERGES = DATA / "merges.txt"


@pytest.fixture(scope="module")
def tokenizers():
    tiktoken = pytest.importorskip("tiktoken")
    if not VOCABULARY.exists() or not MERGES.exists():
        pytest.skip("run scripts/download/gpt2.sh")
    return kael.BpeTokenizer(str(VOCABULARY), str(MERGES)), tiktoken.get_encoding(
        "gpt2"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Hello, my name is",
        "The quick brown fox don't jump, won't they've we'll I'm you're.",
        "  leading and   inner spaces\t\n\n and tabs  \n ",
        "año, 日本語のテキスト, Ελληνικά, русский, عربى 2024€ ٣٤",
        "emoji 🙂👍🏽 and symbols ©®™ — … “quotes”",
        "<|endoftext|> stays plain text",
    ],
)
def test_matches_tiktoken_on_known_text(tokenizers, text):
    ours, reference = tokenizers
    ids = reference.encode_ordinary(text)
    assert ours.encode(text) == ids
    assert ours.decode(ids).decode("utf-8") == text


def test_vocabulary_size_matches_gpt2(tokenizers):
    ours, reference = tokenizers
    assert ours.vocabulary_size == reference.n_vocab


@settings(max_examples=2000, deadline=None)
@given(text=st.text(max_size=120))
def test_matches_tiktoken_on_any_text(tokenizers, text):
    ours, reference = tokenizers
    assert ours.encode(text) == reference.encode_ordinary(text)
