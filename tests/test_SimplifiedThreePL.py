import unittest
#import numpy as np

import unittest
from src.Experiment import Experiment
from src.SimplifiedThreePL import SimplifiedThreePL

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        correct = [55, 60, 75, 90, 95]
        incorrect = [45, 40, 25, 10, 5]
        self.experiment = Experiment(correct, incorrect)
        self.model = SimplifiedThreePL(self.experiment)

    def test_summary(self):
        """Test that summary returns correct values."""
        summary = self.model.summary()
        self.assertEqual(summary["n_conditions"], 5)

    def test_predict_output_range(self):
        """Test that predict() outputs probabilities between 0 and 1."""
        predictions = self.model.predict([1.0, 0.0])
        self.assertTrue(np.all((0 <= predictions) & (predictions <= 1)))

    def test_fit_model(self):
        """Test that fit() successfully updates parameters."""
        self.model.fit()
        self.assertTrue(self.model._is_fitted)

if __name__ == "__main__":
    unittest.main()
#Completed with the help of AI 