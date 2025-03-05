import math
from Experiment import Experiment

class SimplifiedThreePL:
    def __init__(self, experiment):
        """Initialize the SimplifiedThreePL model with an Experiment object."""
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an instance of Experiment")
        self.experiment = experiment
        self._is_fitted = False
        self._discrimination = None
        self._logit_base_rate = None

    def summary(self):
        """Returns experiment summary."""
        return self.experiment.summary()

    def sigmoid(self, x):
        """Computes the sigmoid function (1 / (1 + exp(-x)))."""
        return 1 / (1 + math.exp(-x))

    def predict(self, parameters):
        """Returns the probability of correct responses given parameters."""
        a, logit_c = parameters
        c = self.sigmoid(logit_c)  # Convert logit_c to probability
        probabilities = []
        
        for difficulty in self.experiment.difficulties:
            probability = c + (1 - c) / (1 + math.exp(-a * difficulty))  # No more "self.experiment.ability"
            probabilities.append(probability)
        
        return probabilities

    def negative_log_likelihood(self, parameters):
        """Computes negative log-likelihood for given parameters."""
        p_correct = self.predict(parameters)
        log_likelihood = 0
        
        for i in range(len(self.experiment.correct)):
            correct = self.experiment.correct[i]
            incorrect = self.experiment.incorrect[i]
            
            if p_correct[i] > 0 and (1 - p_correct[i]) > 0:  # Prevent log(0) errors
                log_likelihood += (
                    correct * math.log(p_correct[i]) +
                    incorrect * math.log(1 - p_correct[i])
                )

        return -log_likelihood  # Negative because we minimize

    def fit(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """Performs gradient descent to optimize parameters."""
        a = 1.0  # Initial guess for discrimination parameter
        logit_c = 0.0  # Initial guess for base rate (in logit form)
        
        prev_loss = float("inf")  # Track previous loss for stopping condition
        
        for _ in range(max_iterations):
            p_correct = self.predict([a, logit_c])
            
            # Compute negative log-likelihood for convergence check
            current_loss = self.negative_log_likelihood([a, logit_c])
            
            # Initialize gradients
            dL_da = 0.0
            dL_dlogit_c = 0.0

            for i in range(len(self.experiment.correct)):
                correct = self.experiment.correct[i]
                incorrect = self.experiment.incorrect[i]
                difficulty = self.experiment.difficulties[i]

                prob = p_correct[i]  # Probability of correct response
                if 0 < prob < 1:  # Avoid log(0) errors
                    # Compute gradients using proper partial derivatives
                    dL_da += -(correct - incorrect) * difficulty * prob * (1 - prob)
                    dL_dlogit_c += -(correct - incorrect) * prob * (1 - prob)

            # Update parameters with gradient descent
            a -= learning_rate * dL_da
            logit_c -= learning_rate * dL_dlogit_c

            # Check for convergence
            if abs(prev_loss - current_loss) < tolerance:
                break
            
            prev_loss = current_loss  # Update loss for next iteration

        # Store learned parameters
        self._discrimination = a
        self._logit_base_rate = logit_c
        self._is_fitted = True

    def get_discrimination(self):
        """Returns the estimated discrimination parameter."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self._discrimination

    def get_base_rate(self):
        """Returns the estimated base rate parameter (not logit)."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.sigmoid(self._logit_base_rate)  # Convert from logit to probability

#Completed with the help of AI