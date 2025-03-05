import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment  # Assuming you have an Experiment class

class TestSimplifiedThreePL(unittest.TestCase):
    
    def setUp(self):
        """Set up a mock Experiment object."""
        self.experiment = Experiment()  # Assume Experiment is defined properly
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization_valid(self):
        """Test that constructor handles valid inputs."""
        self.assertIsInstance(self.model, SimplifiedThreePL)

    def test_summary(self):
        """Test that the summary method returns the correct data."""
        summary = self.model.summary()
        self.assertEqual(summary['n_total'], 100)  # Example number of trials

    def test_predict(self):
        """Test that the predict method outputs probabilities between 0 and 1."""
        self.model.fit()  # Fit the model before predicting
        prob = self.model.predict([0.5])  # Example parameter
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
        fitted_ll = self.model.negative_log_likelihood([0.5])
        self.assertLess(fitted_ll, initial_ll)

    def test_get_parameters_before_fit(self):
        """Test that you cannot access parameters before fitting."""
        with self.assertRaises(ValueError):
            self.model.get_discrimination()

    def test_integration(self):
        """Test the integration of the model with a known dataset."""
        # Create mock data with 5 conditions, 100 trials per condition
        self.experiment.set_data_conditions([0.55, 0.60, 0.75, 0.90, 0.95])
        self.model.fit()
        predictions = self.model.predict([0.5])  # Expected behavior
        self.assertEqual(len(predictions), 5)

if __name__ == '__main__':
    unittest.main()
