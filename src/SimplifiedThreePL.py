# Acknowledging referance to and help from some website tools for fixing codes and errors

import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        """Initialize with experiment data."""
        self.experiment = experiment
        self._is_fitted = False

    def summary(self):
        """Summarize the data in the experiment."""
        # Ensure n_correct and n_incorrect are arrays or lists
        n_total = sum(self.experiment.n_correct + self.experiment.n_incorrect)
        n_correct = sum(self.experiment.n_correct)
        n_incorrect = sum(self.experiment.n_incorrect)
        n_conditions = len(self.experiment.n_correct)

        return {
            'n_total': n_total,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'n_conditions': n_conditions
        }

    def predict(self, parameters):
        """
        Predict probabilities based on the parameters for each condition.
        a: discrimination
        theta: person ability
        difficulties: list of item difficulties
        q: base rate adjustment
        """
        a, theta, difficulties, q = parameters
        c = 1 / (1 + np.exp(-q))  # Inverse logit to get c
        probabilities = []
        
        for bi in difficulties:
            prob = c + (1 - c) / (1 + np.exp(-a * (theta - bi)))
            probabilities.append(prob)
        
        return np.array(probabilities)

    def negative_log_likelihood(self, parameters):
        """
        Calculate the negative log-likelihood for the current parameters.
        a: discrimination
        theta: person ability
        difficulties: list of item difficulties
        q: base rate adjustment
        """
        a, theta, difficulties, q = parameters
        c = 1 / (1 + np.exp(-q))  # Inverse logit to get c
        probabilities = self.predict(parameters)
        
        n_correct = self.experiment.n_correct
        n_incorrect = self.experiment.n_incorrect
        
        log_likelihood = 0
        for i in range(len(difficulties)):
            prob_correct = probabilities[i]
            prob_incorrect = 1 - prob_correct
            
            # Update log-likelihood calculation to match your experiment structure
            log_likelihood += (n_correct[i] * np.log(prob_correct)) + (n_incorrect[i] * np.log(prob_incorrect))
        
        return -log_likelihood

    def fit(self):
        """Fit the model using the maximum likelihood estimation."""
        # Initial guess for the parameters: [a, theta, difficulties, q]
        initial_guess = [1.0, 0.0, [2, 1, 0, -1, -2], 0.0]  # Example values for a, theta, difficulties, q
        result = minimize(self.negative_log_likelihood, initial_guess, method='BFGS')

        if result.success:
            # Accessing the optimized parameters from the result
            a, theta, difficulties, q = result.x[0], result.x[1], result.x[2], result.x[3]
            self._discrimination = a
            self._logit_base_rate = q
            self._base_rate = 1 / (1 + np.exp(-q))  # Convert logit to base rate
            self._is_fitted = True
        else:
            raise ValueError("Optimization failed to converge.")

    def get_discrimination(self):
        """Get the fitted discrimination parameter."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        """Get the fitted base rate."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._base_rate
