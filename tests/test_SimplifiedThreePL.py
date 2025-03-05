# Acknowledging referance to and help from some website tools for fixing codes and errors

import sys
import os
# Add the 'src' directory to sys.path to make sure it's in the search path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now import your modules
from SimplifiedThreePL import SimplifiedThreePL
from Experiment import Experiment
from SignalDetection import SignalDetection
import unittest

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        """Set up a mock Experiment object."""
        self.experiment = Experiment()

        # Add some conditions to the experiment
        conditions = [
            (10, 5, 3, 8),  # hits, misses, false alarms, correct rejections
            (10, 7, 2, 8),
            (8, 5, 4, 7),
            (10, 4, 3, 8),
            (7, 3, 2, 5)
        ]
        
        for hits, misses, falseAlarms, correctRejections in conditions:
            sdt = SignalDetection(hits, misses, falseAlarms, correctRejections)
            self.experiment.add_condition(sdt)
        
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization_valid(self):
        """Test that constructor handles valid inputs."""
        self.assertIsInstance(self.model, SimplifiedThreePL)

    def test_summary(self):
        """Test that the summary method returns the correct data."""
        summary = self.model.summary()
        self.assertEqual(summary['n_total'], 55)  # Example sum of trials (adjust if necessary).

    def test_predict(self):
        """Test that the predict method outputs probabilities between 0 and 1."""
        self.model.fit()  # Fit the model before predicting
        parameters = [1.0, 0.5, [2, 1, 0, -1, -2], 0.0]
        prob = self.model.predict(parameters)  
        self.assertTrue(0 <= prob <= 1)

    def test_predict_base_rate_effect(self):
        """Test that higher base rates increase predicted probabilities."""
        self.model.fit()
        prob1 = self.model.predict([0.5])  # With base_rate 0.5
        self.model._base_rate = 0.8  # Increase base rate
        prob2 = self.model.predict([0.5])
        self.assertGreater(prob2, prob1)

    def test_negative_log_likelihood(self):
        """Test that the negative log-likelihood improves after fitting."""
        initial_ll = self.model.negative_log_likelihood([0.5])
        self.model.fit()
        parameters = [1.0, 0.5, [2, 1, 0, -1, -2], 0.0]
        fitted_ll = self.model.negative_log_likelihood([0.5])
        self.model.negative_log_likelihood(parameters)
        self.assertLess(fitted_ll, initial_ll)

    def test_get_parameters_before_fit(self):
        """Test that you cannot access parameters before fitting."""
        with self.assertRaises(ValueError):
            self.model.get_discrimination()

    def test_integration(self):
        """Test the integration of the model with a known dataset."""
        for condition in [0.55, 0.60, 0.75, 0.90, 0.95]:
            sdt = SignalDetection(hits=10, misses=5, falseAlarms=3, correctRejections=8)  # Example data
            self.experiment.add_condition(sdt)
        self.model.fit()
        predictions = self.model.predict([0.5])  # Expected behavior
        self.assertEqual(len(predictions), 5)

if __name__ == '__main__':
    unittest.main()
