import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "workflow", "scripts", "signature")
)

import pandas as pd
import pytest
from signature import signature, d_self, d_ref

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(user_gammas: dict) -> pd.DataFrame:
    """
    user_gammas: {(user, split): list of gamma vectors (one per day)}
    Returns a DataFrame with columns [user, split, date, gamma_0, gamma_1, ...]
    """
    rows = []
    base_date = pd.Timestamp("2023-01-01")
    for (user, split), gammas in user_gammas.items():
        for day_idx, g in enumerate(gammas):
            row = {
                "user": user,
                "split": split,
                "date": base_date + pd.Timedelta(days=day_idx),
            }
            for k, v in enumerate(g):
                row[f"gamma_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# signature() — ranked=False
# ---------------------------------------------------------------------------


class TestSignatureUnranked:
    def test_averages_gamma_per_user_split(self):
        df = _make_df(
            {
                ("u1", "s1"): [[0.8, 0.2], [0.6, 0.4]],
            }
        )
        out = signature(df, ranked=False)
        row = out[(out["user"] == "u1") & (out["split"] == "s1")].iloc[0]
        assert pytest.approx(row["sig_0"], abs=1e-9) == 0.7
        assert pytest.approx(row["sig_1"], abs=1e-9) == 0.3

    def test_user_and_split_are_columns(self):
        df = _make_df({("u1", "s1"): [[1.0, 0.0]]})
        out = signature(df, ranked=False)
        assert "user" in out.columns
        assert "split" in out.columns

    def test_one_row_per_user_split(self):
        df = _make_df(
            {
                ("u1", "s1"): [[0.5, 0.5], [0.5, 0.5]],
                ("u1", "s2"): [[0.9, 0.1]],
                ("u2", "s1"): [[0.3, 0.7]],
            }
        )
        out = signature(df, ranked=False)
        assert len(out) == 3

    def test_single_day_passthrough(self):
        df = _make_df({("u1", "s1"): [[0.3, 0.7]]})
        out = signature(df, ranked=False)
        row = out.iloc[0]
        assert pytest.approx(row["sig_0"]) == 0.3
        assert pytest.approx(row["sig_1"]) == 0.7

    def test_missing_gamma_cols_raises(self):
        df = pd.DataFrame(
            {"user": ["u1"], "split": ["s1"], "date": ["2023-01-01"], "Cluster": [0]}
        )
        with pytest.raises(AssertionError):
            signature(df, ranked=False)

    def test_missing_split_col_raises(self):
        df = pd.DataFrame({"user": ["u1"], "date": ["2023-01-01"], "gamma_0": [1.0]})
        with pytest.raises(AssertionError):
            signature(df, ranked=False)


# ---------------------------------------------------------------------------
# signature() — ranked=True
# ---------------------------------------------------------------------------


class TestSignatureRanked:
    def test_rank1_is_largest_component(self):
        df = _make_df({("u1", "s1"): [[0.2, 0.8]]})
        out = signature(df, ranked=True)
        # rank column names are integers after pivot_table
        assert pytest.approx(out.iloc[0][1]) == 0.8  # rank 1

    def test_ranks_are_descending(self):
        df = _make_df({("u1", "s1"): [[0.1, 0.6, 0.3]]})
        out = signature(df, ranked=True)
        vals = [out.iloc[0][r] for r in range(1, 4)]
        assert vals == sorted(vals, reverse=True)

    def test_user_and_split_present(self):
        df = _make_df({("u1", "s1"): [[0.4, 0.6]]})
        out = signature(df, ranked=True)
        assert "user" in out.columns
        assert "split" in out.columns

    def test_ranked_weights_sum_to_one(self):
        df = _make_df({("u1", "s1"): [[0.25, 0.25, 0.25, 0.25]]})
        out = signature(df, ranked=True)
        rank_cols = [c for c in out.columns if c not in ("user", "split")]
        total = out.iloc[0][rank_cols].sum()
        assert pytest.approx(total) == 1.0


# ---------------------------------------------------------------------------
# d_self()
# ---------------------------------------------------------------------------


class TestDSelf:
    def _sig_df(self):
        # u1 has identical splits → d_self = 0
        # u2 has different splits → d_self > 0
        data = {
            ("u1", "split_1"): [[0.5, 0.5]],
            ("u1", "split_2"): [[0.5, 0.5]],
            ("u2", "split_1"): [[1.0, 0.0]],
            ("u2", "split_2"): [[0.0, 1.0]],
        }
        return signature(_make_df(data), ranked=False)

    def test_identical_splits_give_zero(self):
        sig = self._sig_df()
        out = d_self(sig, splits=["split_1", "split_2"])
        u1_row = out[out["user"] == "u1"]
        assert pytest.approx(u1_row["d_self"].values[0], abs=1e-9) == 0.0

    def test_different_splits_give_positive(self):
        sig = self._sig_df()
        out = d_self(sig, splits=["split_1", "split_2"])
        u2_row = out[out["user"] == "u2"]
        assert u2_row["d_self"].values[0] > 0

    def test_returns_dataframe_with_user_and_d_self(self):
        sig = self._sig_df()
        out = d_self(sig, splits=["split_1", "split_2"])
        assert "user" in out.columns
        assert "d_self" in out.columns


# ---------------------------------------------------------------------------
# d_ref()
# ---------------------------------------------------------------------------


class TestDRef:
    def _sig_df(self):
        data = {
            ("u1", "split_1"): [[1.0, 0.0]],
            ("u1", "split_2"): [[1.0, 0.0]],
            ("u2", "split_1"): [[0.0, 1.0]],
            ("u2", "split_2"): [[0.0, 1.0]],
        }
        return signature(_make_df(data), ranked=False)

    def test_same_signature_users_give_max_distance(self):
        sig = self._sig_df()
        out = d_ref(sig, splits=["split_1", "split_2"], return_="full")
        # u1=[1,0] vs u2=[0,1] are maximally different under JSD → distance = 1
        assert pytest.approx(out.loc["u1", "u2"], abs=1e-6) == 1.0

    def test_full_matrix_is_symmetric(self):
        sig = self._sig_df()
        out = d_ref(sig, splits=["split_1", "split_2"], return_="full")
        assert pytest.approx(out.loc["u1", "u2"]) == out.loc["u2", "u1"]

    def test_condensed_returns_series(self):
        sig = self._sig_df()
        out = d_ref(sig, splits=["split_1", "split_2"], return_="condensed")
        assert isinstance(out, pd.Series)
        assert len(out) == 1  # one pair: (u1, u2)
