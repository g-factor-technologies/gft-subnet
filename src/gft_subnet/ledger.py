"""SQLite ledger for Bittensor windows. Separate from HubStore."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from gft_subnet.protocol import Episode


SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    window_id INTEGER PRIMARY KEY,
    policy_version INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    window_id INTEGER NOT NULL,
    hotkey TEXT NOT NULL,
    family TEXT NOT NULL,
    task_seed INTEGER NOT NULL,
    generation_seed INTEGER NOT NULL,
    policy_version INTEGER NOT NULL,
    reward REAL NOT NULL,
    complete INTEGER NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY(window_id) REFERENCES windows(window_id)
);
CREATE TABLE IF NOT EXISTS miner_acks (
    window_id INTEGER NOT NULL,
    hotkey TEXT NOT NULL,
    checksum TEXT NOT NULL,
    PRIMARY KEY(window_id, hotkey)
);
CREATE TABLE IF NOT EXISTS scores (
    window_id INTEGER NOT NULL,
    hotkey TEXT NOT NULL,
    uid INTEGER NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY(window_id, hotkey)
);
CREATE TABLE IF NOT EXISTS credits (
    window_id INTEGER NOT NULL,
    hotkey TEXT NOT NULL,
    credit REAL NOT NULL,
    failed INTEGER NOT NULL,
    PRIMARY KEY(window_id, hotkey)
);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def open_window(self, policy_version: int, checksum: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO windows(policy_version, checksum) VALUES (?, ?)",
            (policy_version, checksum),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def record_episode(self, window_id: int, episode: Episode) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes(
                episode_id, window_id, hotkey, family, task_seed, generation_seed,
                policy_version, reward, complete, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                window_id,
                episode.hotkey,
                episode.family,
                episode.task_seed,
                episode.generation_seed,
                episode.policy_version,
                float(episode.reward),
                int(episode.complete),
                json.dumps(episode.to_dict()),
            ),
        )
        self._conn.commit()

    def ack(self, window_id: int, hotkey: str, checksum: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO miner_acks(window_id, hotkey, checksum) VALUES (?, ?, ?)",
            (window_id, hotkey, checksum),
        )
        self._conn.commit()

    def acked_hotkeys(self, window_id: int, checksum: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT hotkey FROM miner_acks WHERE window_id = ? AND checksum = ?",
            (window_id, checksum),
        ).fetchall()
        return {str(row["hotkey"]) for row in rows}

    def record_score(self, window_id: int, hotkey: str, uid: int, score: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scores(window_id, hotkey, uid, score) VALUES (?, ?, ?, ?)",
            (window_id, hotkey, uid, float(score)),
        )
        self._conn.commit()

    def episodes_for_window(self, window_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT payload FROM episodes WHERE window_id = ? ORDER BY episode_id",
            (window_id,),
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def scores_for_window(self, window_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT hotkey, uid, score FROM scores WHERE window_id = ? ORDER BY uid",
            (window_id,),
        ).fetchall()
        return [{"hotkey": row["hotkey"], "uid": int(row["uid"]), "score": float(row["score"])} for row in rows]

    def used_tasks(self) -> set[tuple[str, int]]:
        """Every (family, task_seed) already played; a task is single-use."""
        rows = self._conn.execute("SELECT DISTINCT family, task_seed FROM episodes").fetchall()
        return {(str(row["family"]), int(row["task_seed"])) for row in rows}

    def record_credits(
        self, window_id: int, credits: dict[str, float], slashed: set[str],
        warned: set[str] | None = None,
    ) -> None:
        """Flag column: 0 = clean, 1 = TraceError (slashed), 2 = TraceWarning."""
        warned = set(warned or ())
        all_hotkeys = set(credits) | set(slashed) | warned
        rows = [
            (
                window_id,
                hotkey,
                float(credits.get(hotkey, 0.0)),
                1 if hotkey in slashed else (2 if hotkey in warned else 0),
            )
            for hotkey in all_hotkeys
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO credits(window_id, hotkey, credit, failed) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def credit_history(self) -> list[tuple[int, dict[str, float], set[str], set[str]]]:
        """Per window, oldest first: (window_id, credits, slashed hotkeys, warned hotkeys)."""
        history: dict[int, tuple[dict[str, float], set[str], set[str]]] = {}
        for row in self._conn.execute(
                "SELECT window_id, hotkey, credit, failed FROM credits ORDER BY window_id"):
            credits, failed, warned = history.setdefault(int(row["window_id"]), ({}, set(), set()))
            hotkey = str(row["hotkey"])
            flag = int(row["failed"])
            if flag == 1:
                failed.add(hotkey)
            elif flag == 2:
                warned.add(hotkey)
                if float(row["credit"]) > 0.0:
                    credits[hotkey] = float(row["credit"])
            else:
                credits[hotkey] = float(row["credit"])
        return [
            (window_id, credits, failed, warned)
            for window_id, (credits, failed, warned) in history.items()
        ]
