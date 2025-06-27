from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import pydantic_core
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from pydantic_ai.agent import Agent
from typing_extensions import AsyncGenerator

from pydanticai_rag.model.database import DocsSection

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


agent = Agent("openai:gpt-4.1-nano", deps_type=Deps)


async def build_search_db():
    """Build the search database."""
    DOCS_JSON = (
        "https://gist.githubusercontent.com/"
        "samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/"
        "80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json"
    )
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sessions_ta.validate_json(response.content)

    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    async with database_connect(create_db=True, database="pydantic_ai_rag") as pool:
        with logfire.span("create schema"):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for section in sections:
                tg.create_task(insert_doc_section(sem, openai, pool, section))

async def build_articles_db():
    """Build the search database."""
    articles = [
        {
            "title": "The Wonders of Space Exploration",
            "content": "Space exploration has led to numerous discoveries about our universe, from new planets to the origins of life on Earth."
        },
        {
            "title": "The History of Artificial Intelligence",
            "content": "Artificial Intelligence has evolved from simple rule-based systems to complex neural networks capable of learning and reasoning."
        },
        {
            "title": "Benefits of Daily Exercise",
            "content": "Engaging in daily exercise improves mental health, boosts immunity, and increases life expectancy."
        },
        {
            "title": "Understanding Quantum Computing",
            "content": "Quantum computers leverage the principles of quantum mechanics to solve problems much faster than classical computers."
        },
        {
            "title": "Climate Change and Its Impact",
            "content": "Climate change is affecting weather patterns, sea levels, and biodiversity across the globe."
        },
        {
            "title": "The Rise of Electric Vehicles",
            "content": "Electric vehicles are becoming more popular due to advances in battery technology and growing environmental concerns."
        },
        {
            "title": "Basics of Blockchain Technology",
            "content": "Blockchain provides a decentralized ledger for secure and transparent transactions."
        },
        {
            "title": "Healthy Eating Habits",
            "content": "A balanced diet rich in fruits, vegetables, and whole grains is essential for maintaining good health."
        },
        {
            "title": "The Importance of Sleep",
            "content": "Quality sleep is vital for cognitive function, emotional well-being, and physical health."
        },
        {
            "title": "Exploring the Deep Ocean",
            "content": "The deep ocean remains one of the least explored frontiers, home to unique ecosystems and species.",
        },
    ]
    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    async with database_connect(create_db=True, database="articles") as pool:
        with logfire.span("create schema"):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for i, article in enumerate(articles):
                doc_sec = DocsSection(id=i, parent=None, path=f"article_{i}.md", level=0, title=article["title"], content=article["content"])
                tg.create_task(insert_doc_section(sem, openai, pool, doc_sec))


async def insert_doc_section(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval("SELECT 1 FROM doc_sections WHERE url = $1", url)
        if exists:
            logfire.info("Skipping {url=}", url=url)
            return

        with logfire.span("create embedding for {url=}", url=url):
            embedding = await openai.embeddings.create(
                input=section.embedding_content(),
                model="text-embedding-3-small",
            )
        assert len(embedding.data) == 1, (
            f"Expected 1 embedding, got {len(embedding.data)}, doc section: {section}"
        )
        embedding = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            "INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)",
            url,
            section.title,
            section.content,
            embedding_json,
        )


sessions_ta = TypeAdapter(list[DocsSection])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
    database: str = "pydantic_ai_rag",
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn = "postgresql://postgres:postgres@localhost:54320"
    if create_db:
        with logfire.span("check and create DB"):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1", database
                )
                if not db_exists:
                    await conn.execute(f"CREATE DATABASE {database}")
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f"{server_dsn}/{database}")
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == "pydantic":
        asyncio.run(build_search_db())
    elif action == "articles":
        asyncio.run(build_articles_db())
    else:
        sys.exit(1)
