# Dynamic Causal Model for Macroeconometrics
Implementation of the potential outcome time series framework in the context of macroeconomic variables, following the approach presented in  "A nonparametric dynamic causal model for macroeconometric" by Rambashan and Shephard  

# Some pracital considerations

When computing the given estimator, several practical considerations must be taken into account to ensure the accuracy and reliability of the results:

1. **Treatment Observations**: The computation begins by considering observations of the treatment variable \( W \) at time \( t-p \). It's critical to use actual observed data points for the treatment variable rather than interpolated values. This approach maintains the integrity of the causal inference by ensuring that the treatment is anchored in actual events or interventions that occurred at specific points in time, rather than estimates or averages that could introduce bias or misrepresent the temporal dynamics of the treatment effect.

2. **Outcome Observations**: The outcome variable \( Y \) is evaluated at the closest available times \( t \). This method pragmatically adapts to the available data, which may include irregularities such as missing values due to less frequent data release schedules. For example, if the outcome data is not released daily, there might not be an exact match for every \( t \) corresponding to the treatment times. By selecting the closest available outcome data, even if it means slightly predicting into the future, the method accommodates the realities of data collection while still attempting to minimize the temporal discrepancy between the treatment and outcome measurements.

3. **No Lookahead for Regressors**: When finding the closest times for the regressors (treatment and controls), the `find_closest_time_no_lookahead` function is employed to ensure that there is no anticipation of future values. This means that for each regressor at time \( t-p \) or \( t-p-i \) (for lags), we are careful to use the closest past or present value without inadvertently using data from the future. This approach is crucial for preserving the causal interpretation of the analysis, as it avoids introducing forward-looking bias into the regressors.

In practice, this means that the estimator is not simply a theoretical construct but is calculated in a way that reflects the real-world process of data collection, with all its potential gaps and irregularities. This approach enhances the estimator's applicability to actual scenarios, acknowledging that perfect datasets are rare and that analysts often have to make the best use of imperfect data. The distinction in the handling of outcome and regressor variables underscores the nuanced approach required in empirical analysis to balance data availability with methodological rigor.




## Code Usage

#### `compute_estimator` Method
The `compute_estimator` method computes a specific causal estimator. It is based on the following formula:

\[ \text{Estimator} = \frac{1}{T - p} \sum_{t=p+1}^{T} \left( \frac{Y_{t} - Y_{t-p-1}}{F_{t-p}(W_{t-p} + h) - F_{t-p}(W_{t-p} - h)} \right) \times \left( I_{[W_{t-p}, W_{t-p} + h]} - I_{[W'_{t-p}, W'_{t-p} + h]} \right) \]

Here is an example of how to use the `compute_estimator` method:

```python
# Compute the estimator with specified parameters
estimator_value = estimator.compute_estimator(
    p=10,        # Lag period of 10
    h=5,         # Bandwidth of 5
    w=25,        # Treatment value w
    w_prime=-25, # Comparison treatment value w'
    min_obs=10   # Minimum number of observations
)
```

#### `plot_effect` Method
The `plot_effect` method is used to visualize the estimator over a range of lags. Here's an example:

```python
# Plot the effect of the treatment over a range of lags
estimator.plot_effect(
    h=15,                            # Bandwidth parameter
    w=75,                            # Treatment value w
    w_prime=0,                       # Comparison treatment value w'
    p_range=[k for k in range(1, 750, 10)], # Range of lags to consider
    min_obs=10                       # Minimum number of observations
)
```

#### `local_projection` Method
Use `local_projection` to estimate causal effects at specific lags with control variables and lags of treatment and outcome:

```python
beta_estimate, beta_std_error = estimator.local_projection(
    p=200,
    min_obs=50,
    other_controls=["USURTOT Index", "IP CHNG Index"],
    lagW=[30],
    lagY=[0]
)
```

#### `plot_local_projection` Method
`plot_local_projection` visualizes estimated causal effects across different lags:

```python
estimator.plot_local_projection(
    p_range=[k for k in range(1, 750, 10)],
    min_obs=50,
    other_controls=["USURTOT Index", "IP CHNG Index"],
    lagW=[30],
    lagY=[0]
)
```

