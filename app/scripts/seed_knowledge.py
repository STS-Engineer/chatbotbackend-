import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.knowledge import (
    KnowledgeChunk,
    KnowledgeDocument,
    Topic,
    TrainingModule,
)

# Loading .env directly as requested
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=_api_key)


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def build_subject_chunk(subject: dict) -> str:
    details = "\n- ".join(subject.get("details", []))
    alerts = "\n- ".join(subject.get("alerts", []))
    keywords = ", ".join(subject.get("keywords", []))
    formulas = ", ".join(subject.get("formulas", []))
    related = ", ".join(subject.get("related_subject_ids", []))
    tags = ", ".join(subject.get("tags", []))

    return f"""
Subject code: {subject.get("id")}
English title: {subject.get("title_en")}
French title: {subject.get("title_fr")}

Summary:
{subject.get("summary", "")}

Key details:
- {details}

Alerts:
- {alerts}

Keywords:
{keywords}

Formulas:
{formulas}

Related subjects:
{related}

Tags:
{tags}
""".strip()


def build_training_chunk(module: dict) -> str:
    objectives = "\n- ".join(module.get("learning_objectives", []))
    subject_ids = ", ".join(module.get("subject_ids", []))
    return f"""
Training module: {module.get("id")}
Title: {module.get("title")}

Learning objectives:
- {objectives}

Covered subjects:
{subject_ids}
""".strip()


def build_decision_rule_chunk(rule: dict) -> str:
    then_text = "\n- ".join(rule.get("then", []))
    else_text = "\n- ".join(rule.get("else", []))
    related = ", ".join(rule.get("related_subject_ids", []))
    return f"""
Decision rule: {rule.get("id")}
Name: {rule.get("name")}

If:
{rule.get("if", "")}

Then:
- {then_text}

Else:
- {else_text}

Related subjects:
{related}
""".strip()


def build_cause_effect_chunk(item: dict) -> str:
    effects = "\n- ".join(item.get("effects", []))
    drivers = "\n- ".join(item.get("drivers_to_check", []))
    related = ", ".join(item.get("related_subject_ids", []))
    return f"""
Cause-effect: {item.get("id")}
Cause:
{item.get("cause", "")}

Effects:
- {effects}

Drivers to check:
- {drivers}

Related subjects:
{related}
""".strip()


def build_glossary_chunk(item: dict) -> str:
    return f"""
Glossary term: {item.get("term")}
Definition:
{item.get("definition")}
""".strip()


def build_qa_chunk(item: dict) -> str:
    answer_outline = "\n- ".join(item.get("answer_outline", []))
    subject_ids = ", ".join(item.get("subject_ids", []))
    return f"""
Example question:
{item.get("question")}

Answer outline:
- {answer_outline}

Related subjects:
{subject_ids}
""".strip()


def build_question_route_chunk(item: dict) -> str:
    route_subjects = ", ".join(item.get("route_to_subject_ids", []))
    answer_shape = "\n- ".join(item.get("expected_answer_shape", []))
    return f"""
Question pattern:
{item.get("pattern")}

Route to subjects:
{route_subjects}

Expected answer shape:
- {answer_shape}
""".strip()


def add_chunk(
    db: Session,
    *,
    document_id,
    chunk_index: int,
    content: str,
    content_type: str,
    reference_code: str,
    title: str,
    metadata: dict,
) -> int:
    chunk = KnowledgeChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        content_type=content_type,
        reference_code=reference_code,
        title=title,
        extra_data=metadata,
        embedding=embed_text(content),
    )
    db.add(chunk)
    return chunk_index + 1


def main() -> None:
    db: Session = SessionLocal()
    try:
        path = Path(settings.KNOWLEDGE_FILE)
        data = json.loads(path.read_text(encoding="utf-8"))

        # Clear old data for a clean reseed.
        db.query(KnowledgeChunk).delete()
        db.query(TrainingModule).delete()
        db.query(Topic).delete()
        db.query(KnowledgeDocument).delete()
        db.commit()

        doc = KnowledgeDocument(
            title="Costing KMS RAG Knowledge Base",
            source_type="json",
            file_name=path.name,
            domain=data.get("domain"),
            language=", ".join(data.get("language", [])),
            version=data.get("schema_version"),
            extra_data={
                "created_on": data.get("created_on"),
                "updated_on": data.get("updated_on"),
                "purpose": data.get("purpose", []),
                "design_principles": data.get("design_principles", []),
                "global_keywords": data.get("global_keywords", []),
            },
        )
        db.add(doc)
        db.flush()

        chunk_index = 0

        for subject in data.get("subjects", []):
            topic = Topic(
                code=subject["id"],
                title=subject.get("title_en") or subject.get("title_fr"),
                summary=subject.get("summary"),
                details="\n".join(subject.get("details", [])),
                alerts="\n".join(subject.get("alerts", [])),
                examples="\n".join(subject.get("formulas", [])) if subject.get("formulas") else None,
                domain=data.get("domain"),
                extra_data={
                    "title_fr": subject.get("title_fr"),
                    "title_en": subject.get("title_en"),
                    "keywords": subject.get("keywords", []),
                    "related_subject_ids": subject.get("related_subject_ids", []),
                    "tags": subject.get("tags", []),
                    "formulas": subject.get("formulas", []),
                },
            )
            db.add(topic)

            chunk_index = add_chunk(
                db,
                document_id=doc.id,
                chunk_index=chunk_index,
                content=build_subject_chunk(subject),
                content_type="subject",
                reference_code=subject["id"],
                title=subject.get("title_en") or subject.get("title_fr"),
                metadata={
                    "title_fr": subject.get("title_fr"),
                    "title_en": subject.get("title_en"),
                    "summary": subject.get("summary"),
                    "keywords": subject.get("keywords", []),
                    "related_subject_ids": subject.get("related_subject_ids", []),
                    "tags": subject.get("tags", []),
                    "formulas": subject.get("formulas", []),
                },
            )

        for module in data.get("training_modules", []):
            training = TrainingModule(
                code=module["id"],
                title=module["title"],
                objective="\n".join(module.get("learning_objectives", [])),
                content=None,
                difficulty_level="standard",
                domain=data.get("domain"),
                extra_data={"subject_ids": module.get("subject_ids", [])},
            )
            db.add(training)

            chunk_index = add_chunk(
                db,
                document_id=doc.id,
                chunk_index=chunk_index,
                content=build_training_chunk(module),
                content_type="training_module",
                reference_code=module["id"],
                title=module["title"],
                metadata={"subject_ids": module.get("subject_ids", [])},
            )

        for rule in data.get("decision_rules", []):
            chunk_index = add_chunk(
                db,
                document_id=doc.id,
                chunk_index=chunk_index,
                content=build_decision_rule_chunk(rule),
                content_type="decision_rule",
                reference_code=rule["id"],
                title=rule["name"],
                metadata={
                    "related_subject_ids": rule.get("related_subject_ids", []),
                    "if": rule.get("if"),
                },
            )

        for rel in data.get("cause_effect_relationships", []):
            chunk_index = add_chunk(
                db,
                document_id=doc.id,
                chunk_index=chunk_index,
                content=build_cause_effect_chunk(rel),
                content_type="cause_effect",
                reference_code=rel["id"],
                title=rel["cause"],
                metadata={"related_subject_ids": rel.get("related_subject_ids", [])},
            )

        for route in data.get("question_routes", []):
            chunk_index = add_chunk(
                db,
                document_id=doc.id,
                chunk_index=chunk_index,
                content=build_question_route_chunk(route),
                content_type="question_route",
                reference_code=route["pattern"],
                title=route["pattern"],
                metadata={
                    "related_subject_ids": route.get("route_to_subject_ids", []),
                    "expected_answer_shape": route.get("expected_answer_shape", []),
                },
            )

        for qa in data.get("qa_examples", []):
            chunk_index = add_chunk(
                db,
                document_id=doc.id,
                chunk_index=chunk_index,
                content=build_qa_chunk(qa),
                content_type="qa_example",
                reference_code=qa["question"],
                title=qa["question"],
                metadata={"related_subject_ids": qa.get("subject_ids", [])},
            )

        for item in data.get("glossary", []):
            chunk_index = add_chunk(
                db,
                document_id=doc.id,
                chunk_index=chunk_index,
                content=build_glossary_chunk(item),
                content_type="glossary",
                reference_code=item["term"],
                title=item["term"],
                metadata={},
            )

        db.commit()
        print("Knowledge base seeded successfully.")

    finally:
        db.close()


if __name__ == "__main__":
    main()