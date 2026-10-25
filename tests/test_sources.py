import json

from tinygpt.conversation import RawMessage, read_conversations, sessionize, write_conversations
from tinygpt.instagram_parser import fix_mojibake, iter_instagram_conversations
from tinygpt.public_data import detokenize
from tinygpt.sources import iter_telegram_conversations, iter_text_conversations, parse_whatsapp_messages


def test_sessionize_merges_runs_splits_on_gap_and_drops_chats_without_me():
    messages = [
        RawMessage("Friend", 0, "hey"),
        RawMessage("Me", 10, "yo"),
        RawMessage("Me", 20, "wyd"),
        RawMessage("Friend", 10_000_000, "only them talking"),
    ]
    conversations = sessionize(messages, {"me"}, source="test")
    assert len(conversations) == 1
    assert [(t.role, t.text) for t in conversations[0].turns] == [("them", "hey"), ("me", "yo\nwyd")]


def test_jsonl_roundtrip(tmp_path):
    conversations = sessionize([RawMessage("a", 0, "hi"), RawMessage("me", 1, "hello")], {"me"}, source="x")
    path = tmp_path / "c.jsonl"
    write_conversations(path, conversations)
    assert [c.to_json() for c in read_conversations(path)] == [c.to_json() for c in conversations]


def test_whatsapp_android_and_ios_formats_and_multiline():
    android = "12/31/23, 9:41 PM - Friend: u up?\n12/31/23, 9:42 PM - Me: ya\nstill awake\n12/31/23, 9:43 PM - Me: <Media omitted>"
    ios = "[31/12/2023, 21:41:05] Friend: u up?\n[31/12/2023, 21:42:00] Me: ya"
    android_msgs = parse_whatsapp_messages(android)
    assert [(m.sender, m.text) for m in android_msgs] == [("Friend", "u up?"), ("Me", "ya\nstill awake")]
    ios_msgs = parse_whatsapp_messages(ios)
    assert [m.sender for m in ios_msgs] == ["Friend", "Me"]
    assert ios_msgs[1].timestamp - ios_msgs[0].timestamp == 55


def test_telegram_text_entities(tmp_path):
    export = {
        "name": "Friend",
        "messages": [
            {"type": "message", "from": "Friend", "date_unixtime": "100", "text": "check"},
            {"type": "message", "from": "Me", "date_unixtime": "110", "text": ["see ", {"type": "bold", "text": "this"}]},
        ],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(export))
    (conversation,) = iter_telegram_conversations(path, {"Me"})
    assert conversation.turns[-1].text == "see this"


def test_instagram_conversations_fix_mojibake(tmp_path):
    thread_dir = tmp_path / "your_instagram_activity" / "messages" / "inbox" / "friend_1"
    thread_dir.mkdir(parents=True)
    laugh = "😂".encode("utf-8").decode("latin-1")
    thread = {
        "messages": [
            {"sender_name": "Me", "timestamp_ms": 2000, "content": f"lol {laugh}"},
            {"sender_name": "Friend", "timestamp_ms": 1000, "content": "joke"},
            {"sender_name": "Friend", "timestamp_ms": 1500, "content": "Friend sent an attachment."},
        ]
    }
    (thread_dir / "message_1.json").write_text(json.dumps(thread))
    (conversation,) = iter_instagram_conversations(tmp_path, {"Me"})
    assert [(t.role, t.text) for t in conversation.turns] == [("them", "joke"), ("me", "lol 😂")]
    assert fix_mojibake("plain") == "plain"


def test_text_source_splits_paragraphs(tmp_path):
    (tmp_path / "notes.txt").write_text("first thought that is long enough\n\nshort\n\nsecond paragraph, also long enough")
    conversations = list(iter_text_conversations(tmp_path))
    assert [c.turns[0].text for c in conversations] == ["first thought that is long enough", "second paragraph, also long enough"]


def test_detokenize_dailydialog():
    assert detokenize("Good morning , sir . I don ' t know .Can you ?") == "Good morning, sir. I don't know. Can you?"
