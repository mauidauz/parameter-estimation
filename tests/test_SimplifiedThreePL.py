import unittest
import sys
import os

# Add the parent directory to sys.path so Python finds src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import modules correctly
from Experiment import Experiment
from SimplifiedThreePL import SimplifiedThreePL

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
        self.assertTrue(all(0 <= p <= 1 for p in predictions))  # Pure Python check

    def test_fit_model(self):
        """Test that fit() successfully updates parameters."""
        self.model.fit()
        self.assertTrue(self.model._is_fitted)

    def test_invalid_initialization(self):
        """Test that constructor raises an error for mismatched lengths."""
        correct = [55, 60, 75, 90]  # One less than incorrect
        incorrect = [45, 40, 25, 10, 5]
        with self.assertRaises(ValueError):
            Experiment(correct, incorrect)

    def test_higher_base_rate_increases_probabilities(self):
        """Test that increasing base rate results in higher probabilities."""
        p1 = self.model.predict([1.0, -2.0])  # Lower base rate
        p2 = self.model.predict([1.0, 2.0])   # Higher base rate
        self.assertTrue(all(p1[i] < p2[i] for i in range(len(p1))))

    def test_difficulty_affects_probability(self):
        """Test that higher difficulty results in lower probability when a is positive."""
        p1 = self.model.predict([1.0, 0.0])  # Standard difficulty
        p2 = self.model.predict([-1.0, 0.0])  # Negative discrimination
        self.assertTrue(all(p1[i] > p2[i] for i in range(len(p1))))

    def test_higher_ability_increases_probability(self):
        """Test that higher ability results in higher probabilities when a is positive."""
        p1 = self.model.predict([1.0, 0.0])  # Default ability
        p2 = self.model.predict([2.0, 0.0])  # Higher discrimination
        
        self.assertTrue(all(p1[i] < p2[i] for i in range(len(p1))))


    def test_negative_log_likelihood_improves_after_fit(self):
        """Test that negative log-likelihood improves after fitting."""
        initial_loss = self.model.negative_log_likelihood([1.0, 0.0])
        self.model.fit()
        final_loss = self.model.negative_log_likelihood([self.model.get_discrimination(), self.model.get_base_rate()])
        self.assertLess(final_loss, initial_loss)

    def test_steeper_curve_results_in_larger_discrimination(self):
        """Test that a steeper curve results in a larger discrimination parameter."""
        correct_steep = [10, 40, 70, 90, 100]  # More contrast
        incorrect_steep = [90, 60, 30, 10, 0]
        exp_steep = Experiment(correct_steep, incorrect_steep)
        model_steep = SimplifiedThreePL(exp_steep)
        
        # Fit both models before accessing parameters
        self.model.fit()
        model_steep.fit()
        
        self.assertGreater(model_steep.get_discrimination(), self.model.get_discrimination())

    def test_no_parameter_access_before_fitting(self):
        """Test that users cannot access parameters before fitting."""
        with self.assertRaises(RuntimeError):
            self.model.get_discrimination()
        with self.assertRaises(RuntimeError):
            self.model.get_base_rate()

    def test_model_convergence(self):
        """Test that model parameters remain stable when fitting multiple times."""
        self.model.fit()
        a1 = self.model.get_discrimination()
        c1 = self.model.get_base_rate()

        self.model.fit()
        a2 = self.model.get_discrimination()
        c2 = self.model.get_base_rate()

        self.assertAlmostEqual(a1, a2, places=3)
        self.assertAlmostEqual(c1, c2, places=3)

    def test_cannot_create_inconsistent_object(self):
        """Test that users cannot create an inconsistent object."""
        correct = [10, 20, 30]
        incorrect = [5, 5]  # Mismatched length
        with self.assertRaises(ValueError):
            Experiment(correct, incorrect)


if __name__ == "__main__":
    unittest.main()
#Completed with the help of AI 