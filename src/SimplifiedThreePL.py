# Acknowledging reference to and help from ChatGPT for fixing codes and errors

import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        """Initialize with experiment data."""
        self.experiment = experiment
        self._is_fitted = False

    def summary(self):
        """Summarize the data in the experiment."""
        n_total = 0
        n_correct = 0
        n_incorrect = 0
        for sdt in self.experiment.conditions:
            n_total += sdt.n_total_responses()  # Use SignalDetection's method to get total responses
            n_correct += sdt.n_correct_responses()  # Correct responses from SignalDetection
            n_incorrect += sdt.n_incorrect_responses()  # Incorrect responses from SignalDetection
        
        n_conditions = len(self.experiment.conditions)

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

        # Ensure that difficulties is a list or an iterable
        if isinstance(difficulties, list):
            for bi in difficulties:
                prob = c + (1 - c) / (1 + np.exp(-a * (theta - bi)))  # Apply the subtraction to each difficulty
                probabilities.append(prob)
        else:
            # If difficulties isn't a list, handle it as a scalar (though it should be a list in practice)
            prob = c + (1 - c) / (1 + np.exp(-a * (theta - difficulties)))
            probabilities.append(prob)

        return np.array(probabilities)

    def negative_log_likelihood(self, parameters):
        """Calculate the negative log-likelihood for the current parameters.
        a: discrimination
        theta: person ability
        difficulties: list of item difficulties
        q: base rate adjustment
        """
        a, theta, *difficulties, q = parameters
        c = 1 / (1 + np.exp(-q))  # Inverse logit to get c
        probabilities = self.predict([a, theta, difficulties, q])

        # Summing the correct and incorrect responses across all conditions
        n_correct = sum(condition.n_correct_responses() for condition in self.experiment.conditions)
        n_incorrect = sum(condition.n_incorrect_responses() for condition in self.experiment.conditions)

        log_likelihood = 0
        for i in range(len(difficulties)):
            prob_correct = probabilities[i]
            prob_incorrect = 1 - prob_correct

            # Update log-likelihood calculation to match your experiment structure
            log_likelihood += (n_correct * np.log(prob_correct)) + (n_incorrect * np.log(prob_incorrect))

        return -log_likelihood

    def fit(self):
        """Fit the model using the maximum likelihood estimation."""
        # Flatten difficulties and ensure the rest of the parameters are scalar
        initial_guess = [1.0, 0.0] + [2, 1, 0, -1, -2] + [0.0]  # [a, theta, difficulties (flattened), q]
        result = minimize(self.negative_log_likelihood, initial_guess, method='BFGS')

        if result.success:
            # Unpack the optimized parameters correctly
            a, theta, *difficulties, q = result.x  # Unpack all parameters
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