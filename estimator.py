class Estimator:
    def __init__(self, W, Y):
        self.W = W
        self.Y = Y

    def _calculate_F(self, w, h, t_p):
        """
        Calculate the empirical probability F that W is between w-h and w+h
        up to time t-p, using the historical data.
        """
        # Use data only up to t_p (not inclusive) to ensure it's not forward looking
        historical_data = self.W.iloc[:t_p]
        return ((historical_data >= w - h) & (historical_data <= w + h)).mean()

    def estimate(self, w, w_prime, p, h):
        T = len(self.Y)
        # Pre-calculate the empirical probabilities for all t-p to avoid looking ahead
        F_w = [self._calculate_F(w, h, t_p) for t_p in range(p, T)]
        F_w_prime = [self._calculate_F(w_prime, h, t_p) for t_p in range(p, T)]
        
        # Calculate the indicators
        indicator_w = ((self.W >= (w - h)) & (self.W <= (w + h))).astype(int)
        indicator_w_prime = ((self.W >= (w_prime - h)) & (self.W <= (w_prime + h))).astype(int)

        # Calculate the shifted Y values once for efficiency
        Y_shifted = self.Y.shift(p)

        # Nonparametric estimator for tau
        tau_sum = 0
        var_sum = 0
        for t_p in range(p, T):
            tau_numerator = ((self.Y.iloc[t_p] - Y_shifted.iloc[t_p]) * 
                             (indicator_w.iloc[t_p] - indicator_w_prime.iloc[t_p]) / 
                             (F_w[t_p-p] - F_w_prime[t_p-p]))
            tau_sum += tau_numerator
            
            # Variance estimator for tau
            var_numerator = (((self.Y.iloc[t_p] - Y_shifted.iloc[t_p])**2) * 
                             (indicator_w.iloc[t_p] + indicator_w_prime.iloc[t_p]))
            var_sum += var_numerator

        tau_hat = tau_sum / (T - p)
        v_k_squared = var_sum / (T - p)
        var_tau_hat = (h**-1) * v_k_squared / (T - p)

        return tau_hat, var_tau_hat
