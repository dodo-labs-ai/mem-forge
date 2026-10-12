"""Setup shared by the whole test suite."""

import asyncio
import os
from pathlib import Path
import shutil

import pytest

# litellm loads its model cost map when it is first imported. The copy packaged
# with litellm keeps the suite, and every process it starts, off the network.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(scope="session", autouse=True)
def migrated_database_template(tmp_path_factory):
    """Start each new test database from one migrated copy.

    Creating a database applies the whole migration history. The session does
    that once, and a test that opens a path holding no file starts from a copy
    of the result. A test that prepares its own file, such as an earlier schema
    to migrate, opens exactly the file it prepared.
    """
    from memforge.storage.database import Database

    template = tmp_path_factory.mktemp("database-template") / "memforge.db"

    async def build() -> None:
        database = Database(str(template))
        await database.connect()
        await database.close()

    asyncio.run(build())
    connect = Database.connect

    async def connect_from_template(self, *, run_migrations: bool = True) -> None:
        path = Path(self.db_path).expanduser()
        if run_migrations and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, path)
        await connect(self, run_migrations=run_migrations)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Database, "connect", connect_from_template)
        yield
