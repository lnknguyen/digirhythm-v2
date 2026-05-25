import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "workflow", "scripts", "signature")
)

import numpy as np
import pandas as pd
import pytest
from transition_signature import transition_matrix, d_self_transition, d_ref_transition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(user_days: dict) -> pd.DataFrame:
    """
    user_days: {user: list of gamma vectors (one per consecutive day)}
    """
    rows = []
    base_date = pd.Timestamp("2023-01-01")
    for user, gammas in user_days.items():
        for day_idx, g in enumerate(gammas):
            row = {"user": user, "date": base_date + pd.Timedelta(days=day_idx)}
            for k, v in enumerate(g):
                row[f"gamma_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def _uniform_mat(k: int) -> pd.DataFrame:
    """Uniform row-stochastic matrix."""
    data = np.full((k, k), 1.0 / k)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# transition_matrix()
# ---------------------------------------------------------------------------


class TestTransitionMatrix:
    def test_hard_assignments_produce_correct_counts(self):
        # day1 = cluster 0 (certain), day2 = cluster 1 (certain)
        # Expected M: M[0,1] = 1, rest = 0; normalized: row 0 = [0,1]
        df = _make_df({"u1": [[1.0, 0.0], [0.0, 1.0]]})
        mat, _ = transition_matrix(df)
        assert pytest.approx(mat.loc[0, 1]) == 1.0
        assert pytest.approx(mat.loc[0, 0]) == 0.0

    def test_soft_outer_product_accumulation(self):
        # gamma[0] = [0.8, 0.2], gamma[1] = [0.3, 0.7]
        # M = outer([0.8,0.2], [0.3,0.7]) = [[0.24,0.56],[0.06,0.14]]
        # row 0 sum = 0.8, row 1 sum = 0.2
        # normalized row 0 = [0.3, 0.7], row 1 = [0.3, 0.7]
        df = _make_df({"u1": [[0.8, 0.2], [0.3, 0.7]]})
        mat, _ = transition_matrix(df)
        assert pytest.approx(mat.loc[0, 0], abs=1e-6) == 0.3
        assert pytest.approx(mat.loc[0, 1], abs=1e-6) == 0.7
        assert pytest.approx(mat.loc[1, 0], abs=1e-6) == 0.3
        assert pytest.approx(mat.loc[1, 1], abs=1e-6) == 0.7

    def test_rows_sum_to_one_after_normalization(self):
        df = _make_df({"u1": [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2]]})
        mat, _ = transition_matrix(df)
        for i in range(3):
            assert pytest.approx(mat.loc[i].sum(), abs=1e-6) == 1.0

    def test_single_day_returns_zero_matrix(self):
        df = _make_df({"u1": [[0.5, 0.5]]})
        mat, M = transition_matrix(df)
        assert np.allclose(M, 0.0)

    def test_multiple_users_accumulate(self):
        # Both users contribute the same transition: cluster 0 → cluster 1
        df = _make_df(
            {
                "u1": [[1.0, 0.0], [0.0, 1.0]],
                "u2": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
        mat, M = transition_matrix(df, normalize=False)
        # M[0,1] should be 2 (one from each user), M[0,0] = 0
        assert pytest.approx(M[0, 1]) == 2.0
        assert pytest.approx(M[0, 0]) == 0.0

    def test_normalize_false_returns_raw_counts(self):
        df = _make_df({"u1": [[0.8, 0.2], [0.3, 0.7]]})
        _, M = transition_matrix(df, normalize=False)
        expected = np.outer([0.8, 0.2], [0.3, 0.7])
        assert np.allclose(M, expected, atol=1e-9)

    def test_returns_dataframe_with_correct_shape(self):
        df = _make_df({"u1": [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]})
        mat, _ = transition_matrix(df)
        assert mat.shape == (3, 3)


# ---------------------------------------------------------------------------
# d_self_transition()
# ---------------------------------------------------------------------------


class TestDSelfTransition:
    def _identical_mats(self, k=2) -> dict:
        mat = _uniform_mat(k)
        return {"u1": {"split_1": mat, "split_2": mat}}

    def _opposite_mats(self, k=2) -> dict:
        m1 = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]])
        m2 = pd.DataFrame([[0.0, 1.0], [0.0, 1.0]])
        return {"u1": {"split_1": m1, "split_2": m2}}

    def test_identical_mats_give_zero(self):
        out = d_self_transition(self._identical_mats(), splits=["split_1", "split_2"])
        assert pytest.approx(out["d_self"].iloc[0], abs=1e-9) == 0.0

    def test_different_mats_give_positive(self):
        out = d_self_transition(self._opposite_mats(), splits=["split_1", "split_2"])
        assert out["d_self"].iloc[0] > 0

    def test_returns_user_and_d_self_columns(self):
        out = d_self_transition(self._identical_mats(), splits=["split_1", "split_2"])
        assert "user" in out.columns
        assert "d_self" in out.columns

    def test_missing_split_skips_user(self):
        all_mats = {"u1": {"split_1": _uniform_mat(2)}}  # split_2 missing
        out = d_self_transition(all_mats, splits=["split_1", "split_2"])
        assert len(out) == 0


# ---------------------------------------------------------------------------
# d_ref_transition()
# ---------------------------------------------------------------------------


class TestDRefTransition:
    def _mats(self) -> dict:
        m_a = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]])  # always goes to cluster 0
        m_b = pd.DataFrame([[0.0, 1.0], [0.0, 1.0]])  # always goes to cluster 1
        return {
            "u1": {"s1": m_a, "s2": m_a},
            "u2": {"s1": m_b, "s2": m_b},
        }

    def test_maximally_different_users_give_positive_distance(self):
        out = d_ref_transition(self._mats(), splits=["s1", "s2"])
        assert out["d_ref"].iloc[0] > 0

    def test_returns_user_i_user_j_d_ref(self):
        out = d_ref_transition(self._mats(), splits=["s1", "s2"])
        assert {"user_i", "user_j", "d_ref"}.issubset(out.columns)

    def test_one_row_per_pair(self):
        out = d_ref_transition(self._mats(), splits=["s1", "s2"])
        assert len(out) == 1  # one pair: (u1, u2)
