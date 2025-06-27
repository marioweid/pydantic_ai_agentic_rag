from __future__ import annotations as _annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import logfire
import pydantic_core
from openai import AsyncOpenAI
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from typing_extensions import AsyncGenerator
from pydanticai_rag.agents.rag_agent import RagAgent

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


agent = Agent("openai:gpt-4.1-nano", deps_type=Deps)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str: # TODO 
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(
        "create embedding for {search_query=}", search_query=search_query
    ):
        embedding = await context.deps.openai.embeddings.create(
            input=search_query,
            model="text-embedding-3-small",
        )

    assert len(embedding.data) == 1, (
        f"Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}"
    )
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        "SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8",
        embedding_json,
    )
    return "\n\n".join(
        f"# {row['title']}\nDocumentation URL:{row['url']}\n\n{row['content']}\n"
        for row in rows
    )


async def run_agent(question: str): # TODO 
    """Entry point to run the agent and perform RAG based question answering."""
    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    logfire.info('Asking "{question}"', question=question)
    pg_agent = RagAgent(
        "openai:gpt-4.1-nano",
        deps_type=Deps,
        tools=[retrieve],
        description="A RAG agent that can answer questions based on documentation.",
    )

    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
    server_dsn: str = "postgresql://postgres:postgres@localhost:54320",
    database: str = "pydantic_ai_rag",
) -> AsyncGenerator[asyncpg.Pool, None]:
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


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else None
    if question:
        asyncio.run(run_agent(question))
    else:
        print(
            f"Can't process question: {question}",
        )
        sys.exit(1)
