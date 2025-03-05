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
            probability = c + (1 - c) / (1 + math.exp(-a * (0 - difficulty)))
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

    def fit(self, learning_rate=0.01, iterations=1000):
        """Performs gradient descent to optimize parameters."""
        a = 1.0  # Initial guess for discrimination parameter
        logit_c = 0.0  # Initial guess for base rate (in logit form)
        
        for _ in range(iterations):
            # Compute gradients
            p_correct = self.predict([a, logit_c])
            dL_da = sum(
                (self.experiment.correct[i] - self.experiment.incorrect[i]) *
                (-self.experiment.difficulties[i] * p_correct[i] * (1 - p_correct[i]))
                for i in range(len(self.experiment.correct))
            )
            dL_dlogit_c = sum(
                (self.experiment.correct[i] - self.experiment.incorrect[i]) *
                (p_correct[i] * (1 - p_correct[i]))
                for i in range(len(self.experiment.correct))
            )

            # Update parameters
            a -= learning_rate * dL_da
            logit_c -= learning_rate * dL_dlogit_c

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