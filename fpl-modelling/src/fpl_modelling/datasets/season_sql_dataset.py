"""SQL table dataset that keeps history across FPL seasons.

A plain ``pandas.SQLTableDataset`` with ``if_exists: replace`` wipes every
previous season each time the tables are refreshed, and ``if_exists: append``
duplicates rows on every re-run. This dataset upserts by season instead: on
save it deletes only the rows for the season(s) present in the new data, then
appends the new rows, all in a single transaction.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from kedro.io import AbstractDataset, DatasetError
from sqlalchemy import create_engine, inspect, text

logger = logging.getLogger(__name__)


class SeasonSQLTableDataset(AbstractDataset[pd.DataFrame, pd.DataFrame]):
    """Load/save a SQL table, replacing data one season at a time.

    Example catalog entry::

        players_hist:
          type: fpl_modelling.datasets.SeasonSQLTableDataset
          credentials: db_credentials
          table_name: players_hist
          load_args:
            index_col: player_id
    """

    def __init__(
        self,
        *,
        table_name: str,
        credentials: dict[str, Any],
        season_col: str = "season",
        load_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not credentials or "con" not in credentials:
            raise DatasetError("'con' must be provided in credentials.")
        self._table_name = table_name
        self._con = credentials["con"]
        self._season_col = season_col
        self._load_args = load_args or {}
        self.metadata = metadata
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(self._con)
        return self._engine

    def _describe(self) -> dict[str, Any]:
        return {
            "table_name": self._table_name,
            "season_col": self._season_col,
            "load_args": self._load_args,
        }

    def load(self) -> pd.DataFrame:
        return pd.read_sql_table(self._table_name, self.engine, **self._load_args)

    def save(self, data: pd.DataFrame) -> None:
        if self._season_col not in data.columns:
            raise DatasetError(
                f"Cannot save '{self._table_name}': data has no '{self._season_col}' column."
            )
        if data[self._season_col].isna().any():
            raise DatasetError(
                f"Cannot save '{self._table_name}': '{self._season_col}' has missing values."
            )

        seasons = sorted(data[self._season_col].unique().tolist())

        with self.engine.begin() as conn:
            insp = inspect(conn)
            if insp.has_table(self._table_name):
                existing_cols = {c["name"] for c in insp.get_columns(self._table_name)}

                if self._season_col not in existing_cols:
                    # Table predates season tracking: we can't tell which season its
                    # rows belong to, so keep them under a backup name rather than
                    # mixing them in or deleting them.
                    backup = f"{self._table_name}__pre_season_backup"
                    logger.warning(
                        "Table '%s' has no '%s' column; renaming it to '%s' and starting fresh.",
                        self._table_name, self._season_col, backup,
                    )
                    conn.execute(text(f'DROP TABLE IF EXISTS "{backup}"'))
                    conn.execute(text(f'ALTER TABLE "{self._table_name}" RENAME TO "{backup}"'))
                else:
                    # The FPL API adds fields between seasons (e.g. defensive_contribution);
                    # add any new columns so older seasons just get NULLs.
                    for col in data.columns:
                        if col not in existing_cols:
                            conn.execute(
                                text(f'ALTER TABLE "{self._table_name}" ADD COLUMN "{col}"')
                            )
                    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
                    deleted = conn.execute(
                        text(
                            f'DELETE FROM "{self._table_name}" '
                            f'WHERE "{self._season_col}" IN ({placeholders})'
                        ),
                        {f"s{i}": s for i, s in enumerate(seasons)},
                    )
                    logger.info(
                        "Replacing %d existing row(s) for season(s) %s in '%s'.",
                        deleted.rowcount, seasons, self._table_name,
                    )

            data.to_sql(self._table_name, conn, if_exists="append", index=False)

    def _exists(self) -> bool:
        return inspect(self.engine).has_table(self._table_name)
