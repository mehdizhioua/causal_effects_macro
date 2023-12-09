import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



class Estimator:
    def __init__(self, W, Y, treatment = "W_t" , outcome = "Y_t"):
        self.W = W
        self.Y = Y
        self.treatment = treatment
        self.outcome = outcome

    def empirical_probability(self, w, h, t_p):
        """
        :param w: value of the treatment at time t_p
        :param h: bandwidth
        :param t_p: time at which the treatment is observed
        :return: empirical probability of the treatment being in the interval [w-h, w+h] at time t_p
        """
        data_up_to_t_p = self.W.loc[:t_p]
        p_w_h = np.mean((data_up_to_t_p[self.treatment] > w - h) & (data_up_to_t_p[self.treatment] <= w + h))
        return p_w_h

    def cumulative_distribution_function(self, w, t):
        """
        :param w: value of the treatment at time t
        :return: empirical probability of the treatment being less than or equal to w at time t
        """
        data_up_to_t = self.W.loc[:t]
        cdf_w = np.mean(data_up_to_t[self.treatment] <= w)
        return cdf_w

    def indicator_function(self, w, h, t):
        """
        :param w: value of the treatment at time t
        :param h: bandwidth
        :param t: time at which the treatment is observed
        :return: indicator function of the treatment being in the interval [w-h, w+h] at time t
        """
        W_t = self.W.loc[t, self.treatment]
        indicator_w = 1 if (W_t > w - h) and (W_t <= w + h) else 0
        return indicator_w

    def find_closest_time(self, t, y_time_index, y_values):
        """
        Find the closest time index to t for which y_values is not NaN.
        
        :param t: Target time.
        :param y_time_index: Index of the Y dataframe.
        :param y_values: Values of Y corresponding to the y_time_index.
        :return: Closest time index to t that is not NaN.
        """
        # Create a temporary Series with the time index and values
        temp_series = pd.Series(y_values.values, index=y_time_index)

        # Drop all NaN values from the Series
        non_nan_series = temp_series.dropna()

        # If all values are NaN or series is empty, return None
        if non_nan_series.empty:
            print("no value in the serie")
            return None

        # Get the index of the non-NaN value closest to t
        closest_index = non_nan_series.index.get_indexer([t], method='nearest')[0]
        return non_nan_series.index[closest_index]


    def find_closest_time_no_lookahead(self, t, y_time_index, y_values):
        """
        Modified to ensure no lookahead and no NaN values: finds the closest time less than or equal to t.
        """
        # Create a temporary DataFrame with time index and values
        temp_df = pd.DataFrame({'time': y_time_index, 'value': y_values}).set_index('time')

        # Filter for times less than or equal to t and non-NaN values
        valid_times = temp_df.loc[temp_df.index <= t]
        valid_times = valid_times.dropna()

        if valid_times.empty:
            return None

        # Find the closest index to t
        closest_index = valid_times.index.get_indexer([t], method='nearest')[0]
        return valid_times.index[closest_index]

    def compute_sum_element(self, t_p, p, h, w, w_prime):
        """
        :param t_p: time at which the treatment is observed
        :param p: lag
        :param h: bandwidth
        :param w: value of the treatment at time t_p
        :param w_prime: value of the treatment at time t_p - p
        :return: sum element of the estimator and sum element of the variance
        """
        t = t_p + pd.DateOffset(days=p)
        t_p_1 = t_p - pd.DateOffset(days=1)
        
        if t in self.Y.index:
            Y_t = self.Y.loc[t, self.outcome]
        else:
            closest_time_to_t = self.find_closest_time(t, self.Y.index, self.Y[self.outcome])
            Y_t = self.Y.loc[closest_time_to_t, self.outcome]

        if t_p_1 in self.Y.index:
            Y_t_p_1 = self.Y.loc[t_p_1, self.outcome]
        else:
            closest_time_to_t_p_1 = self.find_closest_time(t_p_1, self.Y.index, self.Y[self.outcome])
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
        """
        :param p: lag
        :param h: bandwidth
        :param w: value of the treatment at time t_p
        :param w_prime: value of the treatment at time t_p - p
        :param min_obs: minimum number of observations to compute the estimator
        :return: estimator and variance
        """
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
        """
        :param h: bandwidth
        :param w: value of the treatment at time t_p
        :param w_prime: value of the treatment at time t_p - p
        :param p_range: range of lags
        :param min_obs: minimum number of observations to compute the estimator
        :return: plot of the estimator and its confidence interval as a function of the lag
        """
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

    def local_projection(self, p, other_controls=None, lagW=None, lagY=[0],min_obs=10):
        """
        :param p: lag
        :param other_controls: list of other columns from Y to be used as controls at time t-p
        :param lags: list of integers indicating additional lags of W to use as controls
        :param min_obs: minimum number of observations to compute the estimator
        :return: beta for the treatment and its standard error
        """
        # Prepare the data for regression
        Y_values = []
        W_values = []

        #controls = other columns of Y + lagged W + lagged Y
        control_values = []
        lagged_W_values = []
        lagged_Y_values = []

        for t_p in self.W.index[min_obs:]:
            t = t_p + pd.DateOffset(days=p)
            
            closest_time_to_t = self.find_closest_time(t, self.Y.index, self.Y[self.outcome])
            
            # Get the outcome value at time t
            Y_values.append(self.Y.loc[closest_time_to_t, self.outcome])
  
            # Get the treatment value at time t-p
            W_values.append(self.W.loc[t_p, self.treatment])
            
            # Get control values at time t-p
            if other_controls:
                for control in other_controls:
                    closest_time_to_control = self.find_closest_time_no_lookahead(t_p, self.Y.index, self.Y[control])
                    control_values.append(self.Y.loc[closest_time_to_control, control])

            # Get lagged values of W
            if lagW:
                for lag in lagW:
                    lagged_time = t_p - pd.DateOffset(days=lag)
                    closest_time_to_lagged = self.find_closest_time_no_lookahead(lagged_time, self.W.index, self.W[self.treatment])
                    lagged_W_values.append(self.W.loc[closest_time_to_lagged, self.treatment])
        
            #Get lagged values of Y
            if lagY:
                for lag in lagY:
                    lagged_time = t_p - pd.DateOffset(days=lag)
                    closest_time_to_lagged = self.find_closest_time_no_lookahead(lagged_time, self.Y.index, self.Y[self.outcome])
                    lagged_Y_values.append(self.Y.loc[closest_time_to_lagged, self.outcome])


            #print("time t_p: ", t_p,"time t: ", t, "closest time to t: ", closest_time_to_t)
            #print("Y value: ", self.Y.loc[closest_time_to_t, self.outcome],"W value: ", self.W.loc[t_p, self.treatment],"lagged Y value", self.Y.loc[closest_time_to_lagged, self.outcome])

        # Convert lists to numpy arrays for regression
        Y_values = np.array(Y_values).reshape(-1, 1)
        W_values = np.array(W_values).reshape(-1, 1)

        # Ensure control_values is a 2D array
        if other_controls:
            control_values = np.array(control_values).reshape(-1, len(other_controls))

        # Ensure lagged_W_values is a 2D array
        if lagW:
            lagged_W_values = np.array(lagged_W_values).reshape(-1, len(lagW))

        if lagY:
            lagged_Y_values = np.array(lagged_Y_values).reshape(-1, len(lagY))

        # Concatenate treatment, control, and lagged W values
        if other_controls is not None and lagged_W_values is not None:
            X_values = np.hstack([W_values, control_values, lagged_W_values, lagged_Y_values])
        elif other_controls is not None:
            X_values = np.hstack([W_values, control_values, lagged_Y_values])
        elif lagged_W_values is not None:
            X_values = np.hstack([W_values, lagged_W_values, lagged_Y_values])
        else:
            X_values = np.hstack([W_values, lagged_Y_values])

        column_names = ['W_t_p']  # Name for the treatment variable
        if other_controls:
            column_names.extend(other_controls)  # Add names for other controls
        if lagW:
            column_names.extend([f'W_lag_{lag}' for lag in lagW])  # Add names for lagged W
        if lagY:
            column_names.extend([f'Y_lag_{lag}' for lag in lagY])  # Add names for lagged Y

        X_values_df = pd.DataFrame(X_values, 
                                   columns=column_names)

        # Add a constant to the model for the intercept
        X_values_df = sm.add_constant(X_values_df)

        # Perform the regression to estimate beta
        regression_model = sm.OLS(Y_values, X_values_df)
        regression_results = regression_model.fit()

        # Print the regression summary

        # Extract beta and its standard error
        beta = regression_results.params['W_t_p']  # Extract the coefficient for the treatment variable
        beta_std_error = regression_results.bse['W_t_p']

        return beta, beta_std_error


    def plot_local_projection(self, p_range, other_controls=None, lagW=None, lagY=[0], min_obs=10):
        """
        :param p_range: range of lags
        :param other_controls: list of other columns from Y to be used as controls
        :param lagW: list of integers indicating additional lags of W to use as controls
        :param lagY: list of integers indicating additional lags of Y to use as controls
        :param min_obs: minimum number of observations to compute the estimator
        """
        betas = []
        beta_stds = []
        
        for p in p_range:
            beta, beta_std_error = self.local_projection(p, other_controls, lagW, lagY, min_obs)
            betas.append(beta)
            beta_stds.append(beta_std_error)

        betas = np.array(betas)
        beta_stds = np.array(beta_stds)

        # Calculate the confidence intervals
        upper_bound = betas + 1.96 * beta_stds  # 95% CI
        lower_bound = betas - 1.96 * beta_stds

        plt.figure(figsize=(10, 6))
        plt.plot(p_range, betas, marker='o', label='Estimated Beta')
        plt.fill_between(p_range, lower_bound, upper_bound, color='grey', alpha=0.3, label='95% Confidence Interval')
        plt.xlabel('Lag (p)')
        plt.ylabel('Estimated Beta Coefficient')
        plt.title('Local Projection as a Function of Lag')
        plt.legend()
        plt.show()

    def regress_treatment(self, p, features):
        """
        Regression of W_t on W_{t-p} and specified lags of W and other features.

        :param p: lag
        :param features: dictionary mapping feature names to lists of lags
        :return: regression results
        """
        data = []
        Y_values = [] #should be understood as "target of the regression", not "ouctome Y" !!
        actual_times = []

        for t in self.W.index:
            row = {}
            if "W" in features:
                for lag in features.get("W", []):
                    lagged_time = t - pd.DateOffset(days=p + lag)
                    closest_time_to_lagged = self.find_closest_time_no_lookahead(lagged_time, self.W.index, self.W[self.treatment])
                    row[f'W_lag_{lag}'] = self.W.loc[closest_time_to_lagged, self.treatment]

            # Add lags of other features
            for feature, lags in features.items():
                if feature == "W":
                    continue  # Skip W as it's already handled
                for lag in lags:
                    lagged_time = t - pd.DateOffset(days=p + lag)

                    try:
                        closest_time_to_lagged = self.find_closest_time(lagged_time, self.Y.index, self.Y[feature])
                        row[f'{feature}_lag_{lag}'] = self.Y.loc[closest_time_to_lagged, feature]
                    except:
                        row[f'{feature}_lag_{lag}'] = np.nan

            # Append the row to data if it's complete (no NaNs)
            if not any(pd.isna(val) for val in row.values()):
                data.append(row)
                Y_values.append(self.W.loc[t, self.treatment])
                actual_times.append(t)

        # Convert data to DataFrame for regression
        X_df = pd.DataFrame(data)
        Y_values = np.array(Y_values)

        # Add a constant to the model for the intercept
        X_df = sm.add_constant(X_df)

        # Perform the regression
        regression_model = sm.OLS(Y_values, X_df)
        regression_results = regression_model.fit()

        # Get predicted values and residuals
        predicted_W = regression_results.predict(X_df)

        # Create a DataFrame with actual and predicted values of W

        results_df = pd.DataFrame({
            'Actual_W': Y_values,
            'Predicted_W': predicted_W.values
        }, index=actual_times)

        return regression_results, results_df

