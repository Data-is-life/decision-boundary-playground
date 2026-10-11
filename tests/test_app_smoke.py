"""End-to-end smoke checks; run: python -m unittest discover -s tests -v."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class AppSmokeTests(unittest.TestCase):
    def check_dataset(self, dataset, classes, test_rows, axes, multiclass):
        app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])
        next(widget for widget in app.selectbox if widget.label == "Dataset").set_value(dataset)
        app.run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])

        models = ["Logistic Regression", "SVM (RBF)", "KNN", "Random Forest"]
        comparisons = {"LogReg", "SVM-RBF", "KNN-7", "RF-200"}
        if importlib.util.find_spec("xgboost") is not None:
            models.append("XGBoost")
            comparisons.add("XGB")
        estimator = next(widget for widget in app.selectbox if widget.label == "Estimator")
        self.assertEqual(list(estimator.options), models)

        metric_names = (
            {"Accuracy", "F1 (macro)"} if multiclass
            else {"Accuracy", "Precision", "Recall", "F1", "ROC AUC"}
        )
        for model in models:
            with self.subTest(dataset=dataset, model=model):
                next(widget for widget in app.selectbox if widget.label == "Estimator").set_value(model)
                app.run()
                self.assertFalse(app.exception, [exc.message for exc in app.exception])
                self.assertFalse(app.warning, [warning.value for warning in app.warning])
                self.assertEqual(len(app.dataframe), 3)

                metrics, confusion, comparison = [frame.value for frame in app.dataframe]
                self.assertEqual(set(metrics.index), metric_names)
                self.assertEqual(set(comparison.columns), metric_names)
                self.assertEqual(set(comparison.index), comparisons)
                for values in (metrics.to_numpy(), comparison.to_numpy()):
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue(((values >= 0) & (values <= 1)).all())

                self.assertEqual(list(confusion.index), classes)
                self.assertEqual(list(confusion.columns), classes)
                self.assertEqual(int(confusion.to_numpy().sum()), test_rows)

                charts = app.get("plotly_chart")
                self.assertEqual(len(charts), 1 if multiclass else 2)
                surface = json.loads(charts[0].proto.spec)
                self.assertEqual(surface["layout"]["title"]["text"], f"{dataset} — {model}")
                self.assertEqual(surface["layout"]["xaxis"]["title"]["text"], axes[0])
                self.assertEqual(surface["layout"]["yaxis"]["title"]["text"], axes[1])
                self.assertEqual(
                    {trace["name"] for trace in surface["data"] if trace.get("name", "").startswith("Train ")},
                    {f"Train {label}" for label in classes},
                )
                if multiclass:
                    self.assertTrue(any("ROC is defined for binary tasks" in info.value for info in app.info))
                else:
                    roc = json.loads(charts[1].proto.spec)
                    self.assertEqual(roc["layout"]["title"]["text"], "Receiver Operating Characteristic")

    def test_binary_moons(self):
        self.check_dataset("Moons", ["0", "1"], 180, ("x1", "x2"), multiclass=False)

    def test_multiclass_iris(self):
        self.check_dataset(
            "Iris", ["0", "1", "2"], 45,
            ("Sepal length (cm)", "Sepal width (cm)"), multiclass=True,
        )


if __name__ == "__main__":
    unittest.main()
