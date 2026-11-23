import pandas as pd
import pytest
from kedro.io import DatasetError
from sqlalchemy import create_engine, inspect

from fpl_modelling.datasets import SeasonSQLTableDataset


@pytest.fixture
def con(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


def make_ds(con):
    return SeasonSQLTableDataset(table_name="players_hist", credentials={"con": con})


def gw_rows(season, rounds, points=1):
    return pd.DataFrame(
        {"season": season, "player_id": 1, "round": list(rounds), "total_points": points}
    )


def test_resaving_a_season_replaces_only_that_season(con):
    ds = make_ds(con)
    ds.save(gw_rows("2025-26", range(1, 39)))
    ds.save(gw_rows("2026-27", range(1, 6)))
    # Re-run mid-season with more gameweeks: no duplicates, last season untouched.
    ds.save(gw_rows("2026-27", range(1, 13), points=2))

    df = ds.load()
    counts = df.groupby("season").size().to_dict()
    assert counts == {"2025-26": 38, "2026-27": 12}
    assert (df.loc[df.season == "2025-26", "total_points"] == 1).all()
    assert (df.loc[df.season == "2026-27", "total_points"] == 2).all()


def test_new_api_columns_are_added(con):
    ds = make_ds(con)
    ds.save(gw_rows("2025-26", [1]))
    ds.save(gw_rows("2026-27", [1]).assign(defensive_contribution=3))

    df = ds.load().set_index("season")
    assert df.loc["2026-27", "defensive_contribution"] == 3
    assert pd.isna(df.loc["2025-26", "defensive_contribution"])


def test_legacy_table_without_season_is_backed_up(con):
    engine = create_engine(con)
    gw_rows("x", [1, 2]).drop(columns="season").to_sql("players_hist", engine, index=False)

    make_ds(con).save(gw_rows("2026-27", [1]))

    tables = inspect(engine).get_table_names()
    assert "players_hist__pre_season_backup" in tables
    assert len(pd.read_sql_table("players_hist__pre_season_backup", engine)) == 2
    assert len(make_ds(con).load()) == 1


def test_save_without_season_column_fails(con):
    with pytest.raises(DatasetError, match="no 'season' column"):
        make_ds(con).save(gw_rows("2026-27", [1]).drop(columns="season"))


def test_failed_save_leaves_existing_data(con, monkeypatch):
    ds = make_ds(con)
    ds.save(gw_rows("2026-27", [1, 2, 3]))

    def boom(*args, **kwargs):
        raise RuntimeError("insert failed")

    monkeypatch.setattr(pd.DataFrame, "to_sql", boom)
    with pytest.raises(DatasetError, match="insert failed"):
        ds.save(gw_rows("2026-27", [1]))
    monkeypatch.undo()

    # The delete was rolled back with the failed insert.
    assert len(ds.load()) == 3
