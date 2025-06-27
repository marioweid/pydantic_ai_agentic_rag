from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
import logfire


class PostgressMixin:
    @asynccontextmanager
    async def database_connect(
        self,
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