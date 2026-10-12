# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import os
import sqlite3
from datetime import datetime, timezone
from core.config import ROOT
DB = ROOT / "runtime/telemetry.sqlite3"

def connection():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=5)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS requests(time TEXT, status TEXT, elapsed REAL, tokens INTEGER)")
    return c

def record(status, elapsed, tokens=0):
    c = connection()
    try:
        with c: c.execute("INSERT INTO requests VALUES(?,?,?,?)", (datetime.now(timezone.utc).isoformat(), status, elapsed, tokens))
    finally: c.close()

def summary():
    c = connection()
    try:
        count, seconds, tokens = c.execute("SELECT count(*), avg(elapsed), sum(tokens) FROM requests").fetchone()
        return {"recorded_requests": count, "mean_request_seconds": seconds or 0, "generated_tokens": tokens or 0,
                "host_load_average_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
                "measurement_scope": "API request metadata and OS host load; no chat text logged"}
    finally: c.close()
