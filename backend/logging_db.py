import sqlite3
import json
import time
from typing import List, Optional

DB_PATH = "logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            query TEXT,
            rewritten_query TEXT,
            sources_json TEXT,
            latency_ms REAL,
            num_chunks INTEGER,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_query(query: str, rewritten_query: str, sources: List[str], latency_ms: float, num_chunks: int, error: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO queries (timestamp, query, rewritten_query, sources_json, latency_ms, num_chunks, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (time.time(), query, rewritten_query, json.dumps(sources), latency_ms, num_chunks, error))
    conn.commit()
    conn.close()

def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM queries")
    total_queries = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(latency_ms) FROM queries WHERE error IS NULL")
    avg_latency = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM queries WHERE error IS NOT NULL")
    error_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_queries": total_queries,
        "average_latency_ms": round(avg_latency, 2) if avg_latency else 0.0,
        "error_rate": round(error_count / total_queries, 4) if total_queries > 0 else 0.0
    }
