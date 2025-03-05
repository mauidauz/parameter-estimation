import math

def inverse_normal_cdf(p):
    """Approximates the inverse CDF (percent-point function) of a normal distribution."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1 (exclusive)")

    # Approximation constants
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        """Initialize SignalDetection object with given counts."""
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def hit_rate(self):
        """Returns the hit rate H = hits / (hits + misses)."""
        return self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

    def false_alarm_rate(self):
        """Returns the false alarm rate FA = false alarms / (false alarms + correct rejections)."""
        return self.falseAlarms / (self.falseAlarms + self.correctRejections) if (self.falseAlarms + self.correctRejections) > 0 else 0

    def d_prime(self):
        """Computes d' (sensitivity index) without scipy."""
        H = self.hit_rate()
        FA = self.false_alarm_rate()
        
        # Prevent errors for extreme values (avoid log(0))
        H = max(min(H, 0.9999), 0.0001)
        FA = max(min(FA, 0.9999), 0.0001)

        return inverse_normal_cdf(H) - inverse_normal_cdf(FA)

    def criterion(self):
        """Computes the decision criterion C without scipy."""
        H = self.hit_rate()
        FA = self.false_alarm_rate()
        
        # Prevent errors for extreme values
        H = max(min(H, 0.9999), 0.0001)
        FA = max(min(FA, 0.9999), 0.0001)

        return -0.5 * (inverse_normal_cdf(H) + inverse_normal_cdf(FA))

#Completed with the help of AI
