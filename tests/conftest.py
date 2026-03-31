"""Shared test fixtures for claude-memory."""

from __future__ import annotations

import json

import pytest

from claude_memory.db import MemoryDB


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_memory.db"
    db = MemoryDB(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample session JSONL file."""
    session_id = "test-session-1234"
    filepath = tmp_path / f"{session_id}.jsonl"

    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Let's build a REST API"},
            "timestamp": "2026-03-28T10:00:00Z",
            "cwd": "/tmp/test-project",
            "gitBranch": "main",
            "sessionId": session_id,
            "isMeta": False,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I'll create a REST API. Let's use"
                            " FastAPI for this project because"
                            " it's modern and fast."
                        ),
                    },
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": "/tmp/test-project/main.py",
                            "content": "from fastapi import FastAPI\napp = FastAPI()\n",
                        },
                    },
                ],
            },
            "timestamp": "2026-03-28T10:01:00Z",
            "sessionId": session_id,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Now let me create the tests. TODO: add integration tests later.",
                    },
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": "/tmp/test-project/test_main.py",
                            "content": "import pytest\n",
                        },
                    },
                ],
            },
            "timestamp": "2026-03-28T10:02:00Z",
            "sessionId": session_id,
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "I prefer using pytest over unittest",
            },
            "timestamp": "2026-03-28T10:03:00Z",
            "cwd": "/tmp/test-project",
            "sessionId": session_id,
            "isMeta": False,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "python -m pytest"},
                    },
                ],
            },
            "timestamp": "2026-03-28T10:04:00Z",
            "sessionId": session_id,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "TaskCreate",
                        "input": {
                            "subject": "Add API documentation",
                            "description": "Create OpenAPI docs for all endpoints",
                        },
                    },
                ],
            },
            "timestamp": "2026-03-28T10:05:00Z",
            "sessionId": session_id,
        },
    ]

    with filepath.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    return filepath, session_id
