"""FactTable — structured row extractor + SQLite persistence.

Converts TableData objects from ExtractedDocument into FactRow records and
persists them to a SQLite database at .refinery/facts.db.

Each row in the `facts` table represents one data row from one table in one
document.  Header values are stored as a JSON column so arbitrary-width tables
are supported without schema migrations.

Schema:
    facts(
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id      TEXT NOT NULL,
        table_idx   INTEGER,          -- table index within document
        page        INTEGER,
        row_idx     INTEGER,          -- 0-based row index within table
        caption     TEXT,
        headers     TEXT,             -- JSON array of column header strings
        values      TEXT,             -- JSON array of cell values for this row
        source_file TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    )

Usage:
    ft = FactTable()
    rows = ft.extract(doc)
    ft.persist(rows)
    results = ft.query("SELECT * FROM facts WHERE doc_id=?", (doc_id,))
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.models.extracted_document import ExtractedDocument


# ---------------------------------------------------------------------------
# FactRow schema
# ---------------------------------------------------------------------------

class FactRow(BaseModel):
    """One data row from a structured table."""

    doc_id: str
    table_idx: int         # index of the TableData in doc.tables
    page: int
    row_idx: int           # 0-based row index within the table
    caption: Optional[str]
    headers: list[str]
    values: list[str]      # parallel to headers
    source_file: str


# ---------------------------------------------------------------------------
# FactTable
# ---------------------------------------------------------------------------

class FactTable:
    """Extracts FactRow objects from ExtractedDocuments and persists to SQLite.

    Thread-safe for reads; persist() uses a write lock via SQLite WAL mode.
    """

    DB_PATH = ".refinery/facts.db"
    TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      TEXT    NOT NULL,
            table_idx   INTEGER NOT NULL,
            page        INTEGER NOT NULL,
            row_idx     INTEGER NOT NULL,
            caption     TEXT,
            headers     TEXT    NOT NULL,
            values      TEXT    NOT NULL,
            source_file TEXT,
            created_at  TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);
        CREATE INDEX IF NOT EXISTS idx_facts_page ON facts(page);
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, doc: ExtractedDocument) -> list[FactRow]:
        """Convert all tables in an ExtractedDocument into FactRow objects.

        Skips tables with no rows or no headers.
        Returns FactRow list (not yet persisted).
        """
        rows: list[FactRow] = []

        for t_idx, table in enumerate(doc.tables):
            if not table.headers or not table.rows:
                continue

            for r_idx, data_row in enumerate(table.rows):
                # Pad or truncate values to match header count
                n_cols = len(table.headers)
                padded = list(data_row) + [""] * max(0, n_cols - len(data_row))
                padded = padded[:n_cols]

                rows.append(FactRow(
                    doc_id=doc.doc_id,
                    table_idx=t_idx,
                    page=table.page,
                    row_idx=r_idx,
                    caption=table.caption,
                    headers=table.headers,
                    values=padded,
                    source_file=doc.filename,
                ))

        return rows

    def persist(self, rows: list[FactRow]) -> int:
        """Upsert FactRows into the SQLite database.

        Uses INSERT OR REPLACE based on (doc_id, table_idx, row_idx) uniqueness.
        Returns the number of rows written.
        """
        if not rows:
            return 0

        con = self._connect()
        try:
            # Delete existing rows for these doc_ids first (idempotent re-runs)
            doc_ids = list({r.doc_id for r in rows})
            placeholders = ",".join("?" * len(doc_ids))
            con.execute(
                f"DELETE FROM facts WHERE doc_id IN ({placeholders})", doc_ids
            )

            con.executemany(
                """INSERT INTO facts
                   (doc_id, table_idx, page, row_idx, caption, headers, values, source_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        r.doc_id,
                        r.table_idx,
                        r.page,
                        r.row_idx,
                        r.caption,
                        json.dumps(r.headers),
                        json.dumps(r.values),
                        r.source_file,
                    )
                    for r in rows
                ],
            )
            con.commit()
            return len(rows)
        finally:
            con.close()

    def query(
        self,
        sql: str,
        params: tuple = (),
    ) -> list[dict]:
        """Run an arbitrary read-only SQL query and return rows as dicts.

        Args:
            sql:    SELECT statement.
            params: Optional positional parameters (use ? placeholders).

        Returns:
            List of dicts with column names as keys.
        """
        con = self._connect()
        try:
            con.row_factory = sqlite3.Row
            cur = con.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            con.close()

    def find_by_doc(self, doc_id: str) -> list[FactRow]:
        """Return all FactRows for a given document."""
        rows = self.query(
            "SELECT * FROM facts WHERE doc_id = ? ORDER BY table_idx, row_idx",
            (doc_id,),
        )
        return [self._row_to_fact(r) for r in rows]

    def count(self, doc_id: Optional[str] = None) -> int:
        """Total fact rows in the database, optionally filtered by doc_id."""
        if doc_id:
            rows = self.query(
                "SELECT COUNT(*) AS n FROM facts WHERE doc_id = ?", (doc_id,)
            )
        else:
            rows = self.query("SELECT COUNT(*) AS n FROM facts")
        return rows[0]["n"] if rows else 0

    def doc_ids(self) -> list[str]:
        """Distinct doc_ids that have fact rows."""
        rows = self.query("SELECT DISTINCT doc_id FROM facts ORDER BY doc_id")
        return [r["doc_id"] for r in rows]

    def delete_document(self, doc_id: str) -> None:
        """Remove all fact rows for a document."""
        con = self._connect()
        try:
            con.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.executescript(self.TABLE_DDL)
            con.commit()
        finally:
            con.close()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @staticmethod
    def _row_to_fact(row: dict) -> FactRow:
        return FactRow(
            doc_id=row["doc_id"],
            table_idx=row["table_idx"],
            page=row["page"],
            row_idx=row["row_idx"],
            caption=row.get("caption"),
            headers=json.loads(row["headers"]),
            values=json.loads(row["values"]),
            source_file=row.get("source_file", ""),
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.models.extracted_document import ExtractedDocument

    if len(sys.argv) < 2:
        print("Usage: python -m src.store.fact_table <extracted_doc.json>")
        sys.exit(1)

    doc = ExtractedDocument.model_validate_json(Path(sys.argv[1]).read_text())
    ft = FactTable()
    rows = ft.extract(doc)
    n = ft.persist(rows)
    print(f"Persisted {n} fact rows from {doc.filename} ({len(doc.tables)} tables)")
    print(f"Total rows in DB: {ft.count()}")
    for r in rows[:3]:
        print(f"  p.{r.page} t{r.table_idx} row{r.row_idx}: {r.values}")
