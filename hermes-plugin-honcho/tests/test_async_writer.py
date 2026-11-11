"""Behaviour of the async writer: write order and content, retry, idle cost and shutdown."""

from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

session_mod = importlib.import_module(
    f"{Path(__file__).resolve().parent.parent.name}.session"
)
HonchoSession = session_mod.HonchoSession
HonchoSessionManager = session_mod.HonchoSessionManager

WAIT = 5.0


class FakeSdkSession:
    def __init__(self, client: FakeHoncho, session_id: str) -> None:
        self._client = client
        self.id = session_id

    def add_peers(self, peers: list[Any]) -> None:
        self._client.calls.append(("add_peers", self.id, tuple(p.id for p, _ in peers)))

    def get_peer_configuration(self, peer: Any) -> Any:
        return SimpleNamespace(observe_me=None, observe_others=None)

    def context(self, **kwargs: Any) -> Any:
        return SimpleNamespace(messages=[])

    def add_messages(self, messages: list[tuple[str, str]]) -> None:
        self._client.upload(self.id, messages)


class FakePeer:
    def __init__(self, peer_id: str) -> None:
        self.id = peer_id

    def message(self, content: str) -> tuple[str, str]:
        return (self.id, content)


class FakeHoncho:
    """Records every SDK call; ``fail`` and ``block`` steer add_messages per Honcho session id."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail: dict[str, int] = {}
        self.block: dict[str, threading.Event] = {}
        self.entered: dict[str, threading.Event] = {}

    def peer(self, peer_id: str) -> FakePeer:
        self.calls.append(("peer", peer_id))
        return FakePeer(peer_id)

    def session(self, session_id: str) -> FakeSdkSession:
        self.calls.append(("session", session_id))
        return FakeSdkSession(self, session_id)

    def upload(self, session_id: str, messages: list[tuple[str, str]]) -> None:
        self.entered.setdefault(session_id, threading.Event()).set()
        gate = self.block.get(session_id)
        if gate is not None:
            assert gate.wait(30), "test never released the blocked upload"
        if self.fail.get(session_id, 0) > 0:
            self.fail[session_id] -= 1
            self.calls.append(("add_messages_failed", session_id, tuple(messages)))
            raise RuntimeError("upload failed")
        self.calls.append(("add_messages", session_id, tuple(messages)))

    def uploads(self) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
        return [(c[1], c[2]) for c in self.calls if c[0] == "add_messages"]


class CountingQueue(queue.Queue):
    """Records the timeout of every get() and counts the ones that returned empty-handed."""

    def __init__(self) -> None:
        super().__init__()
        self.get_timeouts: list[float | None] = []
        self.empty_wakeups = 0

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        if block:
            self.get_timeouts.append(timeout)
        try:
            return super().get(block, timeout)
        except queue.Empty:
            if block:
                self.empty_wakeups += 1
            raise


@pytest.fixture
def fake() -> Iterator[FakeHoncho]:
    client = FakeHoncho()
    yield client
    for gate in client.block.values():
        gate.set()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The writer's retry back-off, recorded instead of slept."""
    recorded: list[float] = []
    monkeypatch.setattr(
        session_mod,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=recorded.append),
    )
    return recorded


@pytest.fixture
def manager(
    fake: FakeHoncho, monkeypatch: pytest.MonkeyPatch
) -> Iterator[HonchoSessionManager]:
    monkeypatch.setattr(session_mod, "get_honcho_client", lambda config=None: fake)
    mgr = HonchoSessionManager(honcho=fake)
    yield mgr
    for gate in fake.block.values():
        gate.set()
    mgr.stop_async_writer(timeout=WAIT)


def make_session(n: int, *contents: str) -> HonchoSession:
    session = HonchoSession(
        key=f"cli:{n}",
        user_peer_id="user",
        assistant_peer_id="ai",
        honcho_session_id=f"s{n}",
    )
    for i, content in enumerate(contents):
        session.add_message("user" if i % 2 == 0 else "assistant", content)
    return session


def wait_for(predicate: Any, timeout: float = WAIT) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.01)


def writer(mgr: HonchoSessionManager) -> threading.Thread:
    assert mgr._async_thread is not None
    return mgr._async_thread


def test_async_saves_flush_in_fifo_order_with_the_same_calls(
    manager: HonchoSessionManager, fake: FakeHoncho
) -> None:
    sessions = [make_session(n, f"hello {n}", f"reply {n}") for n in range(5)]
    for session in sessions:
        manager.save(session)
    wait_for(lambda: len(fake.uploads()) == 5)

    expected: list[tuple[Any, ...]] = [("peer", "user"), ("peer", "ai")]
    for n in range(5):
        expected += [
            ("session", f"s{n}"),
            ("add_peers", f"s{n}", ("user", "ai")),
            ("add_messages", f"s{n}", (("user", f"hello {n}"), ("ai", f"reply {n}"))),
        ]
    assert fake.calls == expected
    assert all(m["_synced"] for s in sessions for m in s.messages)


def test_a_failed_batch_is_retried_once(
    manager: HonchoSessionManager,
    fake: FakeHoncho,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake.fail["s0"] = 1
    session = make_session(0, "hello")
    with caplog.at_level(logging.DEBUG, logger=session_mod.__name__):
        manager.save(session)
        wait_for(lambda: len(fake.uploads()) == 1)

    assert [c[0] for c in fake.calls if c[0].startswith("add_messages")] == [
        "add_messages_failed",
        "add_messages",
    ]
    assert sleeps == [2]
    assert session.messages[0]["_synced"] is True
    assert [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ] == [
        (logging.ERROR, "Failed to sync messages to Honcho: upload failed"),
        (logging.WARNING, "Honcho async write failed, retrying once"),
    ]


def test_a_batch_that_fails_twice_is_dropped(
    manager: HonchoSessionManager,
    fake: FakeHoncho,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake.fail["s0"] = 2
    session = make_session(0, "hello")
    with caplog.at_level(logging.DEBUG, logger=session_mod.__name__):
        manager.save(session)
        wait_for(
            lambda: any("dropping batch" in r.getMessage() for r in caplog.records)
        )

    assert [c[0] for c in fake.calls if c[0].startswith("add_messages")] == [
        "add_messages_failed",
        "add_messages_failed",
    ]
    assert sleeps == [2]
    assert session.messages[0]["_synced"] is False
    assert [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ] == [
        (logging.ERROR, "Failed to sync messages to Honcho: upload failed"),
        (logging.WARNING, "Honcho async write failed, retrying once"),
        (logging.ERROR, "Failed to sync messages to Honcho: upload failed"),
        (logging.ERROR, "Honcho async write retry failed, dropping batch"),
    ]


def test_an_idle_writer_never_wakes_up(
    manager: HonchoSessionManager, fake: FakeHoncho
) -> None:
    manager._async_queue = counting = CountingQueue()
    manager.save(make_session(0, "hello"))
    wait_for(lambda: len(fake.uploads()) == 1)
    wait_for(
        lambda: len(counting.get_timeouts) == 2
    )  # the writer is back waiting for the next item

    time.sleep(5.5)  # longer than any periodic wake-up the writer could use

    assert counting.empty_wakeups == 0
    assert counting.get_timeouts == [None, None]


def test_stopping_an_idle_writer_is_prompt(
    manager: HonchoSessionManager, fake: FakeHoncho
) -> None:
    manager.save(make_session(0, "hello"))
    wait_for(lambda: len(fake.uploads()) == 1)

    started = time.monotonic()
    manager.stop_async_writer(timeout=WAIT)

    assert time.monotonic() - started < 1.0
    assert not writer(manager).is_alive()


@pytest.mark.parametrize("stop", ["stop_async_writer", "shutdown"])
def test_a_writer_that_outlives_the_join_still_exits(
    manager: HonchoSessionManager, fake: FakeHoncho, stop: str
) -> None:
    """The join times out while the writer is inside an upload, and the drain that follows takes the shutdown
    sentinel off the queue. The writer must still get it once the upload returns."""
    fake.block["s0"] = release = threading.Event()
    manager.save(make_session(0, "hello"))
    assert fake.entered.setdefault("s0", threading.Event()).wait(WAIT)

    getattr(manager, stop)(timeout=0.2)
    assert writer(manager).is_alive()
    release.set()

    writer(manager).join(timeout=2.0)
    assert not writer(manager).is_alive()
    assert fake.uploads() == [("s0", (("user", "hello"),))]


def test_shutdown_flushes_queued_sessions_inline(
    manager: HonchoSessionManager, fake: FakeHoncho
) -> None:
    fake.block["s0"] = release = threading.Event()
    manager.save(make_session(0, "hello 0"))
    assert fake.entered.setdefault("s0", threading.Event()).wait(WAIT)
    manager.save(make_session(1, "hello 1"))
    manager.save(make_session(2, "hello 2"))
    threading.Timer(0.5, release.set).start()

    manager.shutdown(timeout=WAIT)

    assert fake.uploads() == [
        ("s1", (("user", "hello 1"),)),
        ("s2", (("user", "hello 2"),)),
        ("s0", (("user", "hello 0"),)),
    ]
    assert not writer(manager).is_alive()


def test_shutdown_out_of_time_reports_what_stayed_unsynced(
    manager: HonchoSessionManager, fake: FakeHoncho, caplog: pytest.LogCaptureFixture
) -> None:
    fake.block["s0"] = threading.Event()
    busy = make_session(0, "hello 0")
    manager.save(busy)
    assert fake.entered.setdefault("s0", threading.Event()).wait(WAIT)
    busy.add_message("assistant", "reply 0")
    manager.save(busy)  # waits on the lock the in-flight upload holds
    manager.save(make_session(1, "hello 1"))  # reached after the budget is spent

    with caplog.at_level(logging.WARNING, logger=session_mod.__name__):
        manager.shutdown(timeout=0.3)

    assert fake.uploads() == []
    assert [r.getMessage() for r in caplog.records] == [
        "Honcho flush ran out of time after 0.3s with 3 message(s) in 2 session(s) still unsynced"
    ]
