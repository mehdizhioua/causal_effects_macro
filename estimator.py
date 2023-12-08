import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Estimator:
    def __init__(self, W, Y, treatment = "W_t" , outcome = "Y_t"):
        self.W = W
        self.Y = Y
        self.treatment = treatment
        self.outcome = outcome


    def empirical_probability(self, w, h, t_p):
        data_up_to_t_p = self.W.loc[:t_p]
        p_w_h = np.mean((data_up_to_t_p[self.treatment] > w - h) & (data_up_to_t_p[self.treatment] <= w + h))
        return p_w_h

    def cumulative_distribution_function(self, w, t):
        data_up_to_t = self.W.loc[:t]
        cdf_w = np.mean(data_up_to_t[self.treatment] <= w)
        return cdf_w

    def indicator_function(self, w, h, t):
        W_t = self.W.loc[t, self.treatment]
        indicator_w = 1 if (W_t > w - h) and (W_t <= w + h) else 0
        return indicator_w

    def find_closest_time(self, t, y_time_index):
        closest_index = y_time_index.get_indexer([t], method='nearest')[0]
        return y_time_index[closest_index]

    def compute_sum_element(self, t_p, p, h, w, w_prime):
        t = t_p + pd.DateOffset(days=p)
        t_p_1 = t_p - pd.DateOffset(days=1)
        
        if t in self.Y.index:
            Y_t = self.Y.loc[t, self.outcome]
        else:
            closest_time_to_t = self.find_closest_time(t, self.Y.index)
            Y_t = self.Y.loc[closest_time_to_t, self.outcome]

        if t_p_1 in self.Y.index:
            Y_t_p_1 = self.Y.loc[t_p_1, self.outcome]
        else:
            closest_time_to_t_p_1 = self.find_closest_time(t_p_1, self.Y.index)
            Y_t_p_1 = self.Y.loc[closest_time_to_t_p_1, self.outcome]

        p_w_h = self.empirical_probability(self.W.loc[t_p, self.treatment], h, t_p)
        indicator_w = self.indicator_function(w, h, t_p)
        indicator_w_prime = self.indicator_function(w_prime, h, t_p)

        if p_w_h > 0:
            sum_element_tau = (Y_t - Y_t_p_1) * (indicator_w - indicator_w_prime) / p_w_h
            sum_element_var = ((Y_t - Y_t_p_1)**2) * (indicator_w + indicator_w_prime) / (p_w_h**2)
        else:
            sum_element_tau = np.nan
            sum_element_var = np.nan
        return sum_element_tau, sum_element_var

    def compute_estimator(self, p, h, w, w_prime, min_obs=10):
        sum_elements_tau = []
        sum_elements_var = []
        for t_p in self.W.index[min_obs:]:
            sum_element_tau, sum_element_var = self.compute_sum_element(t_p, p, h, w, w_prime)
            sum_elements_tau.append(sum_element_tau)
            sum_elements_var.append(sum_element_var)

        tau = np.nanmean(sum_elements_tau)
        var = np.nanmean(sum_elements_var)
        return tau, var

    def plot_effect(self, h, w, w_prime, p_range, min_obs=10):
        estimators = []
        stds = []
        for p in p_range:
            estimator,var= self.compute_estimator(p, h, w, w_prime, min_obs)
            estimators.append(estimator)
            stds.append(np.sqrt(var))

        estimators = np.array(estimators)
        stds = np.array(stds)

        plt.figure(figsize=(10, 6))
        plt.plot(p_range, estimators, marker='o', label='Estimator')
        plt.plot(p_range, estimators + stds, linestyle='--', color='grey', label='Estimator + 1 STD')
        plt.plot(p_range, estimators - stds, linestyle='--', color='grey', label='Estimator - 1 STD')
        plt.xlabel('Lag (p)')
        plt.ylabel('Estimated Value')
        plt.title('Estimator Values as a Function of Lag with Confidence Intervals')
        plt.legend()
        plt.show()

