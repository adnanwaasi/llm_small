import torch

from tinygpt.bpe import ByteBPETokenizer
from tinygpt.chat_format import build_reply_prompt, encode_conversation
from tinygpt.config import ModelConfig, TrainingConfig
from tinygpt.conversation import Conversation, Turn
from tinygpt.dataset import IGNORE_INDEX, ChatStream
from tinygpt.model import TinyGPT
from tinygpt.trainer import learning_rate_at, load_checkpoint, train_model

TOKENIZER = ByteBPETokenizer.train(["hey wyd\n", "nothing much u?\n", "same lol\n"] * 10, vocab_size=280)
CONVERSATION = Conversation(turns=[Turn("them", "hey wyd"), Turn("me", "nothing much u?"), Turn("them", "same")])


def test_loss_mask_only_covers_my_turns():
    ids, mask = encode_conversation(TOKENIZER, CONVERSATION, train_on="me")
    learned = [token for token, flag in zip(ids, mask) if flag]
    assert learned == TOKENIZER.encode("nothing much u?") + [TOKENIZER.end_id]

    _, all_mask = encode_conversation(TOKENIZER, CONVERSATION, train_on="all")
    assert all_mask[0] == 0 and all(all_mask[1:])


def test_reply_prompt_ends_with_open_me_turn():
    prompt = build_reply_prompt(TOKENIZER, [Turn("them", "hey")])
    assert prompt[0] == TOKENIZER.bos_id and prompt[-1] == TOKENIZER.me_id


def test_batches_always_contain_trainable_targets():
    stream = ChatStream.from_conversations(TOKENIZER, [CONVERSATION] * 20, train_on="me")
    inputs, targets = stream.get_batch(16, 8, torch.device("cpu"))
    assert inputs.shape == targets.shape == (16, 8)
    assert ((targets != IGNORE_INDEX).sum(dim=1) > 0).all()


def test_model_forward_ignores_masked_targets_and_generate_stops():
    config = ModelConfig(vocab_size=TOKENIZER.vocab_size, context_length=16, embedding_dim=32, num_heads=2, num_layers=1)
    model = TinyGPT(config)
    inputs = torch.randint(0, config.vocab_size, (2, 16))
    _, loss = model(inputs, torch.full((2, 16), IGNORE_INDEX).index_fill(1, torch.tensor([3]), 5))
    assert torch.isfinite(loss)
    assert model.language_model_head.weight is model.token_embedding.weight

    model.eval()
    out = model.generate(inputs[:1, :4], 40, stop_token_ids=set(range(config.vocab_size)))
    assert out.shape[1] == 5  # stopped after exactly one token
    long = model.generate(inputs[:1, :4], 30, top_p=0.5, banned_token_ids={TOKENIZER.end_id})
    assert TOKENIZER.end_id not in long[0, 4:].tolist()


def test_lr_schedule_warms_up_then_decays():
    config = TrainingConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100)
    assert learning_rate_at(0, config) < learning_rate_at(9, config) == 1e-3
    assert abs(learning_rate_at(99, config) - 1e-4) < 1e-5


def test_pretrain_then_finetune_end_to_end(tmp_path):
    model_config = ModelConfig(vocab_size=TOKENIZER.vocab_size, context_length=16, embedding_dim=32, num_heads=2, num_layers=1)
    stream = ChatStream.from_conversations(TOKENIZER, [CONVERSATION] * 30, train_on="all")
    config = TrainingConfig(run_name="pre", max_steps=30, warmup_steps=5, eval_interval=15, eval_batches=2,
                            batch_size=4, device="cpu", output_dir=str(tmp_path))
    best = train_model(config, model_config, TOKENIZER, stream, stream)
    model, tokenizer, loaded_config, payload = load_checkpoint(best, "cpu")
    assert tokenizer.to_dict() == TOKENIZER.to_dict() and loaded_config == model_config

    me_stream = ChatStream.from_conversations(TOKENIZER, [CONVERSATION] * 30, train_on="me")
    fine = TrainingConfig(stage="finetune", run_name="fine", max_steps=40, warmup_steps=2, eval_interval=10,
                          eval_batches=2, batch_size=4, device="cpu", output_dir=str(tmp_path), learning_rate=3e-3, patience=2)
    best_fine = train_model(fine, model_config, TOKENIZER, me_stream, me_stream, init_state=model.state_dict())
    assert torch.load(best_fine, weights_only=False)["val_loss"] < payload["val_loss"] + 1.0
