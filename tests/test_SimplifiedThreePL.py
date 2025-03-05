# Acknowledging reference to and help from some website tools for fixing codes and errors

import sys
import os
import numpy as np 

# Adding the 'src' directory to sys.path to make sure it's in the search path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importing modules
from SimplifiedThreePL import SimplifiedThreePL #src. ? but src. isnt working 
from Experiment import Experiment #src. ?
from SignalDetection import SignalDetection
import unittest

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        """Set up a mock Experiment object."""
        self.experiment = Experiment()

        # Add some conditions to the experiment
        conditions = [
            (10, 5, 3, 8),  # order: hits, misses, false alarms, correct rejections
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

    def test_predict(self):
        """Test that the predict method outputs probabilities between 0 and 1."""
        self.model.fit()  # Fit the model before predicting
        parameters = [1.0, 0.5, [2, 1, 0, -1, -2], 0.0]  # Correct number of parameters
        prob = self.model.predict(parameters)  
        self.assertTrue(np.all(prob >= 0) and np.all(prob <= 1))  # Ensures probabilities are between 0 and 1

    def test_predict_base_rate_effect(self):
        """Test that higher base rates increase predicted probabilities."""
        self.model.fit()
        prob1 = self.model.predict([1.0, 0.5, [2, 1, 0, -1, -2], 0.0])  # Correct parameters
        self.model._base_rate = 0.8  # Increase base rate
        prob2 = self.model.predict([1.0, 0.5, [2, 1, 0, -1, -2], 0.0])  # Correct parameters
        # Use a small tolerance to account for rounding errors
        self.assertGreater(prob2[0], prob1[0] + 1e-5)  # Increased tolerance


    def test_negative_log_likelihood(self):
        """Test that the negative log-likelihood improves after fitting."""
        initial_ll = self.model.negative_log_likelihood([1.0, 0.5, [2, 1, 0, -1, -2], 0.0])  # Correct parameters
        self.model.fit()
        final_ll = self.model.negative_log_likelihood([1.0, 0.5, [2, 1, 0, -1, -2], 0.0])  # Correct parameters
        self.assertLess(final_ll, initial_ll)  # Log-likelihood should improve after fitting

if __name__ == "__main__":
    unittest.main()