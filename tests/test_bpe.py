from tinygpt.bpe import ByteBPETokenizer
from tinygpt.tokenizer import load_tokenizer, save_tokenizer

CORPUS = ["lol idk man\n", "wanna grab food later?\n", "haan bhai kal milte hai 😂\n", "lol same\n"] * 20


def test_roundtrip_handles_emoji_hinglish_and_unseen_text():
    tokenizer = ByteBPETokenizer.train(CORPUS, vocab_size=300)
    for text in ["lol idk man", "kal milte hai 😂", "totally unseen wörds 🤖\nnew line"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_merges_compress_frequent_text():
    tokenizer = ByteBPETokenizer.train(CORPUS, vocab_size=320)
    assert len(tokenizer.encode("lol idk man")) < len("lol idk man".encode())


def test_special_tokens_sit_after_vocab_and_are_skipped_on_decode():
    tokenizer = ByteBPETokenizer.train(CORPUS, vocab_size=300)
    assert tokenizer.vocab_size == 300
    assert max(tokenizer.special_ids.values()) == tokenizer.vocab_size - 1
    ids = [tokenizer.me_id] + tokenizer.encode("hey") + [tokenizer.end_id]
    assert tokenizer.decode(ids) == "hey"
    assert tokenizer.decode(ids, skip_special=False) == "<|me|>hey<|end|>"
    assert tokenizer.end_id not in tokenizer.encode("<|end|>")


def test_save_and_load(tmp_path):
    tokenizer = ByteBPETokenizer.train(CORPUS, vocab_size=300)
    path = tmp_path / "tok.json"
    save_tokenizer(tokenizer, path)
    loaded = load_tokenizer(path)
    assert loaded.encode("lol same 😂") == tokenizer.encode("lol same 😂")
    assert loaded.special_ids == tokenizer.special_ids
