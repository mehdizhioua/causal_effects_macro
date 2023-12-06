import pandas as pd

class Estimator:
    def __init__(self, W, Y):
        self.W = W
        self.Y = Y

    def estimate(self, w, w_prime, p, h):
        T = len(self.Y)
        indicator_w = ((self.W >= (w - h)) & (self.W <= (w + h))).astype(int)
        indicator_w_prime = ((self.W >= (w_prime - h)) & (self.W <= (w_prime + h))).astype(int)

        # Calculate Ft-p for each t
        Ft_p = self.W.rolling(window=p).mean().shift(-p)
        
        # Calculate ft-p(W_t-p) for each t
        ft_p = self.W.diff(periods=p).shift(-p)
        
        # Nonparametric estimator for tau
        tau_numerator = ((self.Y - self.Y.shift(p)) * (indicator_w - indicator_w_prime) /
                         (Ft_p + h - Ft_p - h)).sum()
        tau_hat = (1 / (T - p)) * tau_numerator

        # Variance estimator for tau
        var_numerator = (((self.Y - self.Y.shift(p))**2) * (indicator_w + indicator_w_prime) /
                         ft_p**2).sum()
        v_k_squared = (1 / (T - p)) * var_numerator
        var_tau_hat = (h**-1) * (v_k_squared / (T - p))

        return tau_hat, var_tau_hat

# Example usage:
# Assume we have dataframes W_df and Y_df corresponding to W_t and Y_t
# W_df = pd.DataFrame(...)
# Y_df = pd.DataFrame(...)

# Initialize the estimator with dataframes
# estimator = Estimator(W_df, Y_df)

# Calculate the estimator and variance
# tau_hat, var_tau_hat = estimator.estimate(w=..., w_prime=..., p=..., h=...)

# Since we don't have actual data to work with, we cannot run this code here.
