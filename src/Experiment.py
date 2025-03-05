class Experiment:
    def __init__(self, correct, incorrect, difficulties=None):
        """Initialize an Experiment with correct/incorrect responses for each condition."""
        self.correct = list(correct)  # Convert input to standard Python lists
        self.incorrect = list(incorrect)

        # Default difficulties if none are provided
        if difficulties is None:
            self.difficulties = [2.0, 1.0, 0.0, -1.0, -2.0]
        else:
            self.difficulties = list(difficulties)

        # Ensure all lists have the same length
        if len(self.correct) != len(self.incorrect) or len(self.correct) != len(self.difficulties):
            raise ValueError("All input lists must have the same length.")

    def total_responses(self):
        """Returns the total number of responses."""
        return sum(self.correct) + sum(self.incorrect)

    def summary(self):
        """Returns a dictionary summarizing the experiment data."""
        return {
            "n_total": self.total_responses(),
            "n_correct": sum(self.correct),
            "n_incorrect": sum(self.incorrect),
            "n_conditions": len(self.correct)
        }

# Developed with help from AI