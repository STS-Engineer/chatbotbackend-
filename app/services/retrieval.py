import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.config import settings

# Loading .env directly as requested
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=_api_key)


def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def _get_training_subject_ids(db: Session, training_code: str) -> list[str]:
    sql = text("""
        SELECT metadata
        FROM training_modules
        WHERE code = :training_code
        LIMIT 1
    """)
    row = db.execute(sql, {"training_code": training_code}).fetchone()
    if not row:
        return []

    metadata = row[0] or {}
    subject_ids = metadata.get("subject_ids", [])
    return [str(x) for x in subject_ids if x]


def _compute_boost(
    chunk: dict[str, Any],
    mode: str,
    topic_code: str | None = None,
    training_code: str | None = None,
    subject_ids: list[str] | None = None,
) -> float:
    content_type = chunk.get("content_type")
    reference_code = chunk.get("reference_code")
    metadata = chunk.get("metadata") or {}
    related_subject_ids = metadata.get("related_subject_ids", []) or []

    boost = 0.0

    if mode == "topic" and topic_code:
        if reference_code == topic_code:
            boost += 0.30
        elif topic_code in related_subject_ids:
            if content_type == "subject":
                boost += 0.15
            elif content_type in {"decision_rule", "cause_effect", "qa_example", "question_route"}:
                boost += 0.12
            elif content_type == "glossary":
                boost += 0.03

    elif mode == "training" and training_code:
        if reference_code == training_code:
            boost += 0.30
        elif subject_ids and reference_code in subject_ids:
            boost += 0.15
        elif subject_ids and any(x in subject_ids for x in related_subject_ids):
            if content_type in {"decision_rule", "cause_effect", "qa_example", "question_route"}:
                boost += 0.12
            elif content_type == "glossary":
                boost += 0.03

    else:
        type_boosts = {
            "subject": 0.08,
            "decision_rule": 0.10,
            "cause_effect": 0.09,
            "qa_example": 0.05,
            "question_route": 0.04,
            "training_module": 0.02,
            "glossary": 0.01,
        }
        boost += type_boosts.get(content_type, 0.0)

    return boost


def retrieve_relevant_chunks(
    db: Session,
    question: str,
    mode: str,
    topic_code: str | None = None,
    training_code: str | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    query_embedding = embed_query(question)
    subject_ids: list[str] = []

    if mode == "training" and training_code:
        subject_ids = _get_training_subject_ids(db, training_code)

    sql = text("""
        SELECT
            reference_code,
            title,
            content,
            content_type,
            metadata,
            (embedding <=> CAST(:embedding AS vector)) AS distance
        FROM knowledge_chunks
        WHERE content_type IN (
            'subject',
            'training_module',
            'decision_rule',
            'cause_effect',
            'glossary',
            'qa_example',
            'question_route'
        )
        ORDER BY distance ASC
        LIMIT 20
    """)

    rows = db.execute(sql, {"embedding": str(query_embedding)}).fetchall()
    candidates = [dict(r._mapping) for r in rows]

    rescored: list[dict[str, Any]] = []
    for chunk in candidates:
        # Distance might be None if the chunk has no embedding in the DB
        raw_dist = chunk.get("distance")
        distance = float(raw_dist) if raw_dist is not None else 1.0
        similarity = 1.0 - distance

        boost = _compute_boost(
            chunk=chunk,
            mode=mode,
            topic_code=topic_code,
            training_code=training_code,
            subject_ids=subject_ids,
        )

        final_score = similarity + boost
        chunk["final_score"] = final_score
        rescored.append(chunk)

    # SORT by final_score descending
    rescored.sort(key=lambda x: x["final_score"], reverse=True)

    seen_codes: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for chunk in rescored:
        code = chunk.get("reference_code")
        if code and code in seen_codes:
            continue

        if code:
            seen_codes.add(code)

        deduped.append(chunk)

        if len(deduped) >= limit:
            break

    return deduped