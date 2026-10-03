"""Query-time retrieval over the sqlite-vec index.

Retrieval only decides WHICH procedure and WHICH diagram(s) are relevant. The exact
facts (torque, ranges, inventory) are resolved later via deterministic tools, not
vector search, which is what keeps this out of "basic RAG" territory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlite_vec import serialize_float32

from .. import config
from ..data_loader import Corpus, load_corpus
from ..models import embedder
from .build_index import connect


@dataclass
class RetrievedProcedure:
    procedure_id: str
    title: str
    score: float


@dataclass
class RetrievedDiagram:
    diagram_id: str
    title: str
    image_path: str
    image_exists: bool
    score: float


@dataclass
class RetrievalResult:
    query: str
    procedures: list[RetrievedProcedure] = field(default_factory=list)
    diagrams: list[RetrievedDiagram] = field(default_factory=list)

    @property
    def top_procedure_id(self) -> str | None:
        return self.procedures[0].procedure_id if self.procedures else None


class Retriever:
    """Loads the corpus once and serves fast vector searches."""

    def __init__(self, db_path=None):
        self.db_path = db_path or config.VECTOR_DB_PATH
        self.corpus: Corpus = load_corpus()

    def _search(self, query: str) -> list[tuple]:
        qvec = embedder.embed_query(query)
        conn = connect(self.db_path)
        try:
            _ensure_index_ready(conn, self.db_path)
            total = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0] or 0
            # sqlite-vec applies the KNN `k` BEFORE any extra WHERE filter, so filtering
            # by doc_type inside the query can starve one type (e.g. return 0 procedures
            # if the nearest rows are all diagrams). Fetch the full ranking, then filter
            # and cap in Python. The corpus is tiny, so this is cheap and correct.
            rows = conn.execute(
                """
                SELECT d.doc_type, d.ref_id, d.title, d.image_path, v.distance
                FROM vec_documents v
                JOIN documents d ON d.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance;
                """,
                (serialize_float32(qvec.tolist()), max(total, 1)),
            ).fetchall()
        finally:
            conn.close()
        return rows

    def retrieve(
        self,
        query: str,
        k_procedures: int | None = None,
        k_diagrams: int = 2,
        include_linked_diagrams: bool = True,
    ) -> RetrievalResult:
        k_proc = k_procedures or config.RETRIEVAL_TOP_K
        result = RetrievalResult(query=query)

        rows = self._search(query)
        for _dt, ref_id, title, _img, dist in (r for r in rows if r[0] == "procedure"):
            if len(result.procedures) >= k_proc:
                break
            result.procedures.append(
                RetrievedProcedure(ref_id, title, _distance_to_score(dist))
            )

        diagram_hits: dict[str, RetrievedDiagram] = {}
        for _dt, ref_id, title, image_path, dist in (r for r in rows if r[0] == "diagram"):
            if len(diagram_hits) >= k_diagrams:
                break
            diagram_hits[ref_id] = RetrievedDiagram(
                diagram_id=ref_id,
                title=title,
                image_path=image_path or "",
                image_exists=bool(image_path and Path(image_path).exists()),
                score=_distance_to_score(dist),
            )

        # Also include diagrams structurally linked to the top procedure, so a step's
        # verify.visual_ref is always available even if it did not match by text.
        if include_linked_diagrams and result.top_procedure_id:
            proc = self.corpus.procedures.get(result.top_procedure_id)
            if proc:
                for diagram_id in proc.related_diagrams:
                    if diagram_id in diagram_hits:
                        continue
                    diagram = self.corpus.diagrams.get(diagram_id)
                    if diagram:
                        diagram_hits[diagram_id] = RetrievedDiagram(
                            diagram_id=diagram_id,
                            title=diagram.title,
                            image_path=str(diagram.image_path),
                            image_exists=diagram.image_exists,
                            score=0.0,  # linked, not similarity-ranked
                        )

        result.diagrams = list(diagram_hits.values())
        return result


def _distance_to_score(distance: float) -> float:
    """Convert L2/cosine distance to a rough 0..1 similarity for display."""
    return round(1.0 / (1.0 + float(distance)), 4)


def _ensure_index_ready(conn, db_path) -> None:
    """Fail with a clear, actionable message if the vector index isn't built yet.

    Without this, sqlite auto-creates an empty DB file and retrieval crashes deep in a
    'no such table' error. This surfaces the real fix instead.
    """
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view');"
        ).fetchall()
    }
    if "documents" not in tables or "vec_documents" not in tables:
        raise RuntimeError(
            f"Vector index not found at {db_path}. Build it first with: "
            "python -m backend.index.build_index"
        )


def index_is_ready(db_path=None) -> bool:
    """Non-raising readiness check (used by the API /ready endpoint)."""
    path = Path(db_path or config.VECTOR_DB_PATH)
    if not path.exists():
        return False
    try:
        conn = connect(path)
        try:
            _ensure_index_ready(conn, path)
            return conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0] > 0
        finally:
            conn.close()
    except Exception:
        return False
