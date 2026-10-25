import tempfile
import unittest
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from spotify_analysis.data import FEATURE_COLUMNS, NUMERIC_COLUMNS, load_data
from spotify_analysis.preprocessing import scale_split, oversample


def features(values):
    table = pd.DataFrame({c: values for c in NUMERIC_COLUMNS})
    table["key"] = 0
    table["mode"] = 1
    table["explicit"] = 0
    return table[FEATURE_COLUMNS]


class PreprocessingTests(unittest.TestCase):
    def test_validation_outlier_does_not_change_training_parameters(self):
        train, validation = features([1., 3.]), features([101.])
        original = train.copy()
        scaled_train, scaled_validation, scaler = scale_split(train, validation)
        self.assertAlmostEqual(scaler.mean_[0], 2.)
        self.assertAlmostEqual(scaled_validation.iloc[0].danceability, 99.)
        self.assertAlmostEqual(scaled_train.danceability.mean(), 0.)
        assert_frame_equal(train, original)

    def test_scaling_preserves_categories_and_indices(self):
        train = features([1., 3.]); train["key"] = [2, 10]
        validation = features([5.]); validation.index = [91]
        scaled_train, scaled_validation, _ = scale_split(train, validation)
        assert_series_equal(scaled_train["key"], train["key"])
        self.assertEqual(scaled_validation.index.tolist(), [91])

    def test_oversampling_retains_originals_without_new_feature_values(self):
        train = features([1., 2., 3., 9.])
        target = pd.Series([0, 0, 0, 1], name="popularity")
        original_train, original_target = train.copy(), target.copy()
        balanced, labels = oversample(train, target)
        self.assertEqual(labels.value_counts().to_dict(), {0: 3, 1: 3})
        self.assertEqual(sorted(balanced.danceability.unique()), [1., 2., 3., 9.])
        self.assertTrue((balanced.loc[labels == 1, "danceability"] == 9).all())
        assert_frame_equal(train, original_train)
        assert_series_equal(target, original_target)

    def test_misaligned_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            oversample(features([1., 2.]), pd.Series([0, 1], index=[5, 6]))

    def test_resampling_is_reproducible(self):
        train = features([1., 2., 3., 8., 9.])
        target = pd.Series([0, 0, 0, 1, 1], name="popularity")
        a, b = oversample(train, target, seed=42)
        c, d = oversample(train, target, seed=42)
        assert_frame_equal(a, c); assert_series_equal(b, d)


class DataTests(unittest.TestCase):
    def fixture(self):
        table = features([1., 2., 3., 4.])
        table["popularity"] = [0, 0, 1, 1]
        table["track_name"] = ["a", "b", "c", "d"]
        table["track_artist"] = "artist"
        table["track_id"] = ["id1", "id2", "id3", "id4"]
        return table

    def test_target_and_identifiers_are_excluded_from_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"songs.csv"; self.fixture().to_csv(path, index=False)
            _, X, y = load_data(path)
            self.assertEqual(X.columns.tolist(), FEATURE_COLUMNS)
            self.assertNotIn("popularity", X)
            self.assertNotIn("track_id", X)
            self.assertTrue(X.index.equals(y.index))

    def test_missing_values_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"songs.csv"; table = self.fixture()
            table.loc[0, "energy"] = float("nan"); table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing values"):
                load_data(path)

    def test_missing_source_has_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "data/README.md"):
            load_data("/nonexistent/spotify-analysis-test-input.csv")


if __name__ == "__main__":
    unittest.main()
