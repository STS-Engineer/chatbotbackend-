from sqlalchemy import Column, Text, Integer, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.core.database import Base
import uuid


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    source_type = Column(Text, nullable=False)
    file_name = Column(Text)
    domain = Column(Text)
    language = Column(Text)
    version = Column(Text)
    extra_data = Column("metadata", JSONB)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_documents.id", ondelete="CASCADE"))
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
    content_type = Column(Text, nullable=False)
    reference_code = Column(Text)
    title = Column(Text)
    extra_data = Column("metadata", JSONB)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Topic(Base):
    __tablename__ = "topics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    summary = Column(Text)
    details = Column(Text)
    alerts = Column(Text)
    examples = Column(Text)
    domain = Column(Text)
    extra_data = Column("metadata", JSONB)
    created_at = Column(TIMESTAMP, server_default=func.now())


class TrainingModule(Base):
    __tablename__ = "training_modules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    objective = Column(Text)
    content = Column(Text)
    difficulty_level = Column(Text)
    domain = Column(Text)
    extra_data = Column("metadata", JSONB)
    created_at = Column(TIMESTAMP, server_default=func.now())