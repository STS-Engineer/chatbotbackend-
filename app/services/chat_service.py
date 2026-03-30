import os
from pathlib import Path
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from sqlalchemy.orm import Session
from app.core.config import settings
from app.services.retrieval import retrieve_relevant_chunks

# Loading .env directly as requested
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

_api_key = os.getenv("OPENAI_API_KEY", "")

# Use AsyncOpenAI for streaming support
client = AsyncOpenAI(api_key=_api_key)


class ChatGenerationError(Exception):
    pass


def build_system_prompt(
    mode: str,
    topic_code: str | None = None,
    training_code: str | None = None,
) -> str:
    base = """
You are a professional costing RAG assistant.

Guidelines:
1.For general greetings or simple inputs (e.g., "hello", "hi", "who are you?"), respond in a polite, friendly, and professional manner as a helpful assistant, keeping the reply clear and concise while offering help if needed.
2. For technical costing questions:
   - Answer only using the provided knowledge context.
   - Do not use outside knowledge or invent advice.
   - If the context does not clearly support a technical answer, say:
     "I do not have enough grounded context in the knowledge base to answer this precisely."
   - Keep the answer tied to the retrieved source content.
   - Be practical, clear, and business-oriented.

Technical Answer format (if technical):
1. Direct answer
2. Key rules
3. Risks / alerts
4. Business impacts
5. Related concepts

Additional instructions:
- If a decision_rule chunk is present, state it explicitly.
- If a cause_effect chunk is present, include the impacts explicitly.
- Prefer the most relevant chunks; do not use all chunks equally.
- Keep the answer tied to the retrieved source content.
- Be practical, clear, and business-oriented.
- Do not mention unrelated topics.

Formatting instructions:
- Use Markdown or HTML for answers to make them visually clear and engaging.
- Use headings, bullet points, numbered lists, and tables for structure.
- Use **bold text** for emphasis.
- You may use simple HTML styling (e.g., <span style="color:#2E86C1;">text</span>) for highlighting key elements.
- Keep formatting clean, professional, and readable.
- Prefer concise and structured outputs.
- If explaining a process, always use numbered steps or tables.
""".strip()

    if mode == "overview":
        base += "\n\nThe user is in overview mode. Answer from the most relevant retrieved chunks in the knowledge base."
    elif mode == "topic" and topic_code:
        base += f"\n\nThe user is in topic mode for {topic_code}. Prioritize that topic and directly related chunks."
    elif mode == "training" and training_code:
        base += f"\n\nThe user is in training mode for {training_code}. Stay within module scope and linked subjects."

    return base


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No grounded context was retrieved."

    parts: list[str] = []

    for c in chunks:
        metadata = c.get("metadata") or {}
        related_subject_ids = metadata.get("related_subject_ids", [])

        parts.append(
            "\n".join(
                [
                    f"Type: {c.get('content_type', 'unknown')}",
                    f"Code: {c.get('reference_code', 'N/A')}",
                    f"Title: {c.get('title', 'Untitled')}",
                    f"Related subjects: {', '.join(related_subject_ids) if related_subject_ids else 'None'}",
                    f"Content: {c.get('content', '')}",
                ]
            )
        )

    return "\n\n---\n\n".join(parts)


async def generate_answer(
    db: Session,
    message: str,
    mode: str,
    topic_code: str | None = None,
    training_code: str | None = None,
):
    """
    Async generator that yields text chunks (and hides metadata from the stream).
    """
    try:
        # Note: retrieve_relevant_chunks is currently synchronous. 
        # For a truly async flow, we would make it async too.
        chunks = retrieve_relevant_chunks(
            db=db,
            question=message,
            mode=mode,
            topic_code=topic_code,
            training_code=training_code,
            limit=4,
        )

        context = _format_context(chunks)
        
        response = await client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": build_system_prompt(mode, topic_code, training_code),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question:\n{message}\n\n"
                        f"Answer only from the context above using the strict 5-point format requested."
                    ),
                },
            ],
            temperature=0.3,
            stream=True,
        )

        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                # Remove the marker if it somehow appears in the chunk
                if "[[HAS_CONTEXT]]" in content:
                    content = content.replace("[[HAS_CONTEXT]]", "").strip()
                if content:
                    yield content

    except (APIConnectionError, RateLimitError) as exc:
        yield f"Error: OpenAI service issue ({exc})"
    except APIStatusError as exc:
        yield f"Error: OpenAI returned status {exc.status_code}."
    except Exception as exc:
        yield f"Error: An unexpected error occurred: {exc}"