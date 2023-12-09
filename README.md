# Dynamic Causal Model for Macroeconometrics
Implementation of the potential outcome time series framework in the context of macroeconomic variables, following the approach presented in  "A nonparametric dynamic causal model for macroeconometric" by Rambashan and Shephard  

# Some pracital considerations

When computing the given estimator, several practical considerations must be taken into account to ensure the accuracy and reliability of the results:

1. **Treatment Observations**: The computation begins by considering observations of the treatment variable \( W \) at time \( t-p \). It's critical to use actual observed data points for the treatment variable rather than interpolated values. This approach maintains the integrity of the causal inference by ensuring that the treatment is anchored in actual events or interventions that occurred at specific points in time, rather than estimates or averages that could introduce bias or misrepresent the temporal dynamics of the treatment effect.

2. **Outcome Observations**: The outcome variable \( Y \) is evaluated at the closest available times \( t \). This method pragmatically adapts to the available data, which may include irregularities such as missing values due to less frequent data release schedules. For example, if the outcome data is not released daily, there might not be an exact match for every \( t \) corresponding to the treatment times. By selecting the closest available outcome data, even if it means slightly predicting into the future, the method accommodates the realities of data collection while still attempting to minimize the temporal discrepancy between the treatment and outcome measurements.

3. **No Lookahead for Regressors**: When finding the closest times for the regressors (treatment and controls), the `find_closest_time_no_lookahead` function is employed to ensure that there is no anticipation of future values. This means that for each regressor at time \( t-p \) or \( t-p-i \) (for lags), we are careful to use the closest past or present value without inadvertently using data from the future. This approach is crucial for preserving the causal interpretation of the analysis, as it avoids introducing forward-looking bias into the regressors.

In practice, this means that the estimator is not simply a theoretical construct but is calculated in a way that reflects the real-world process of data collection, with all its potential gaps and irregularities. This approach enhances the estimator's applicability to actual scenarios, acknowledging that perfect datasets are rare and that analysts often have to make the best use of imperfect data. The distinction in the handling of outcome and regressor variables underscores the nuanced approach required in empirical analysis to balance data availability with methodological rigor.



# Example Usage of `local_projection`

The `local_projection` method can be used to estimate the causal effect of a treatment variable at a specific lag while controlling for other variables and additional lags of both the treatment and outcome variables. Here's an example of how to use this method:

```python
# Estimating the local projection at a specific lag
beta_estimate, beta_std_error = estimator.local_projection(
    p=200,  # Lag of 200 days
    min_obs=50,  # Minimum number of observations
    other_controls=["USURTOT Index", "IP CHNG Index"],  # Other control variables
    lagW=[30],  # Additional 30-day lag of the treatment variable
    lagY=[0]  # Include the current value (lag 0) of the outcome variable
)
```

- `p=200`: Specifies the lag (in days) at which the treatment effect is estimated.
- `min_obs=50`: Sets the minimum number of observations required to compute the estimator.
- `other_controls=["USURTOT Index", "IP CHNG Index"]`: Includes additional control variables in the model.
- `lagW=[30]`: Adds a 30-day lag of the treatment variable as an additional control.
- `lagY=[0]`: Includes the current value (lag 0) of the outcome variable as a control.

### Example Usage of `plot_local_projection`

The `plot_local_projection` method is used to visualize how the estimated causal effect changes across a range of lags. Here's an example of how to use this method:

```python
# Plotting the local projection across a range of lags
estimator.plot_local_projection(
    p_range=[k for k in range(1, 750, 10)],  # Range of lags from 1 to 750 with a step of 10
    min_obs=50,  # Minimum number of observations
    other_controls=["USURTOT Index", "IP CHNG Index"],  # Other control variables
    lagW=[30],  # Additional 30-day lag of the treatment variable
    lagY=[0]  # Include the current value (lag 0) of the outcome variable
)
```

- `p_range=[k for k in range(1, 750, 10)]`: Defines a range of lags to explore, from 1 to 750 days, in steps of 10 days.
- The other arguments (`min_obs`, `other_controls`, `lagW`, `lagY`) are similar to those in `local_projection`.

---

These examples provide a clear guide on how to utilize the `local_projection` and `plot_local_projection` methods in practical scenarios, showcasing the flexibility of the Estimator class in handling various types of controls and lags for comprehensive causal analysis.
