"""What the function-hook module relies on: the task read from the transcript (or a task it sends), and the X-Winnow header."""

import json
import threading
from urllib.request import Request, urlopen

import pytest

from winnow import serve
from winnow.hooks import Runtime
from winnow.transcript import find_transcript, guess_transcript_path, task_from_payload, transcript_for


def numbered(n):
    return "\n".join(f"line {i}" for i in range(1, n + 1))


def payload(tool_use_id="t1", **extra):
    return {
        "hook_event_name": "PostToolUse",
        "source": "function-hook",
        "session_id": "s",
        "tool_use_id": tool_use_id,
        "tool_name": "Bash",
        "tool_input": {"command": "cat big.log"},
        "tool_response": {"stdout": numbered(100), "stderr": "", "interrupted": False, "isImage": False},
        "task": {"user_request": "find the failing step", "assistant_intent": "reading the log"},
        **extra,
    }


# --------------------------------------------------------------------------- transcript helpers


def test_task_override_beats_the_transcript():
    task = task_from_payload(payload())
    assert task is not None
    assert task.user_request == "find the failing step"
    assert task.assistant_intent == "reading the log"


def test_task_override_ignores_junk():
    assert task_from_payload({}) is None
    assert task_from_payload({"task": "not a dict"}) is None
    assert task_from_payload({"task": {"user_request": "  ", "assistant_intent": ""}}) is None


def test_task_override_trims_like_the_transcript_reader():
    long = "x" * 3000
    task = task_from_payload({"task": {"user_request": long, "assistant_intent": long}}, max_chars=100)
    assert task is not None
    assert len(task.user_request) == 100 and task.user_request.endswith("…")
    assert len(task.assistant_intent) == 100 and task.assistant_intent.startswith("…")


def test_guessed_transcript_path_uses_claude_codes_slug(tmp_path, monkeypatch):
    monkeypatch.setenv("WINNOW_TRANSCRIPTS_ROOT", str(tmp_path))
    path = guess_transcript_path(r"C:\Work\my app", "abc-123")
    assert path == str(tmp_path / "C--Work-my-app" / "abc-123.jsonl")


def test_transcript_for_falls_back_to_the_guess(tmp_path, monkeypatch):
    monkeypatch.setenv("WINNOW_TRANSCRIPTS_ROOT", str(tmp_path))
    main = tmp_path / "C--Work-app" / "s9.jsonl"
    main.parent.mkdir(parents=True)
    main.write_text("{}\n", encoding="utf-8")
    assert transcript_for({"session_id": "s9", "cwd": r"C:\Work\app"}) == str(main)
    # and the subagent rule still applies on top of the guess
    sub = main.parent / "s9" / "subagents" / "agent-a1.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text("{}\n", encoding="utf-8")
    assert transcript_for({"session_id": "s9", "cwd": r"C:\Work\app", "agent_id": "a1"}) == str(sub)
    assert transcript_for({"session_id": "s9"}) is None


# --------------------------------------------------------------------------- sidecar answers


@pytest.fixture
def server(cfg, fake_judge_cls):
    judge = fake_judge_cls({"b001": 0.9, "b002": 0.0, "b003": 0.0, "b004": 0.9, "error_present": 0.0})
    srv = serve.make_server(0, runtime=Runtime(cfg, judge, None))
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield srv, judge
    srv.shutdown()
    srv.server_close()


def post(srv, path, obj):
    port = srv.server_address[1]
    req = Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as resp:
        body = resp.read()
        return resp.status, (json.loads(body) if body else None), resp.headers.get("X-Winnow")


def test_the_modules_task_reaches_the_judge(server):
    srv, judge = server
    status, body, _ = post(srv, "/hook/post-tool-use", payload("t1"))
    assert status == 200 and body is not None
    assert len(judge.calls) == 1
    assert judge.calls[0][0]["task"] == {"user_request": "find the failing step", "assistant_intent": "reading the log"}


def test_x_winnow_header_summarises_the_rewrite(server):
    srv, _ = server
    status, body, header = post(srv, "/hook/post-tool-use", payload("t3"))
    meta = json.loads(header)
    assert meta["hidden"] == 2 and meta["blocks"] == 4
    assert meta["before"] > meta["after"] > 0
    assert meta["key"] and meta["key"] in body["hookSpecificOutput"]["updatedToolOutput"]["stdout"]


def test_pass_through_has_no_header(server):
    srv, _ = server
    small = payload("t4")
    small["tool_response"]["stdout"] = "tiny"
    status, body, header = post(srv, "/hook/post-tool-use", small)
    assert status == 200 and body is None and header is None


def test_decisions_record_the_source(server, cfg):
    from winnow import log

    srv, _ = server
    post(srv, "/hook/post-tool-use", payload("t7"))
    events = [e for e in log.read_events(cfg) if e.get("tool_use_id") == "t7"]
    assert events and events[-1]["source"] == "function-hook"


def test_a_session_that_changed_directory_still_finds_its_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("WINNOW_TRANSCRIPTS_ROOT", str(tmp_path))
    main = tmp_path / "C--Work-app" / "s10.jsonl"
    main.parent.mkdir(parents=True)
    main.write_text("{}\n", encoding="utf-8")
    (tmp_path / "C--Work-other").mkdir()
    # Started in C:\Work\app, working in a subdirectory now: the guess misses and the search finds it.
    assert transcript_for({"session_id": "s10", "cwd": r"C:\Work\app\sidecar"}) == str(main)
    sub = main.parent / "s10" / "subagents" / "agent-a2.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text("{}\n", encoding="utf-8")
    assert transcript_for({"session_id": "s10", "cwd": r"C:\Work\app\sidecar", "agent_id": "a2"}) == str(sub)


def test_the_search_takes_only_a_plain_session_id(tmp_path, monkeypatch):
    monkeypatch.setenv("WINNOW_TRANSCRIPTS_ROOT", str(tmp_path / "projects"))
    (tmp_path / "projects" / "p").mkdir(parents=True)
    (tmp_path / "secret.jsonl").write_text("{}\n", encoding="utf-8")
    assert find_transcript("../../secret") is None
    assert find_transcript("..") is None
    assert find_transcript("nobody") is None


def test_without_a_task_the_judge_reads_it_from_the_transcript(server, tmp_path, monkeypatch):
    """What the module sends since 0.5.2: no task, so the sidecar reads the transcript's tail."""
    monkeypatch.setenv("WINNOW_TRANSCRIPTS_ROOT", str(tmp_path / "projects"))
    transcript = tmp_path / "projects" / "C--Work-app" / "s11.jsonl"
    transcript.parent.mkdir(parents=True)
    rows = [
        {"type": "user", "message": {"role": "user", "content": "Find why the nightly build fails"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Reading the build log."}]}},
    ]
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    srv, judge = server
    body = payload("t9", session_id="s11", cwd=r"C:\Work\app\ci")
    del body["task"]
    status, _, _ = post(srv, "/hook/post-tool-use", body)
    assert status == 200 and len(judge.calls) == 1
    assert judge.calls[0][0]["task"] == {"user_request": "Find why the nightly build fails", "assistant_intent": "Reading the build log."}
