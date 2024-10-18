import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import (
    ARIMA, ExponentialSmoothing, LinearRegressionModel, 
    Theta, TCNModel, NBEATSModel, RandomForest, NaiveSeasonal, NaiveDrift, RegressionEnsembleModel, RNNModel
)
from darts.utils.missing_values import fill_missing_values
from darts.utils.timeseries_generation import sine_timeseries
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.explainability.shap_explainer import ShapExplainer
from darts.utils.likelihood_models import QuantileRegression
import shap

# Helper function for plotting
def plot_series(series, title="Time Series"):
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Streamlit App Layout
st.title("Darts Time Series Forecasting Dashboard")

# Load the AirPassengers dataset
series = AirPassengersDataset().load()

st.header("1. Data Exploration")
st.write("### Air Passengers Dataset")
plot_series(series, title="Air Passengers Over Time")

# Scaling Data
st.write("### Scaled Air Passengers Data")
scaler = Scaler()
scaled_series = scaler.fit_transform(series)
plot_series(scaled_series, title="Scaled Air Passengers Data")

st.header("2. Forecasting Models")

# Model selection
model_option = st.selectbox(
    "Choose a model to apply:",
    ("ARIMA", "ExponentialSmoothing", "Linear Regression", "TCN (Deep Learning)", "N-BEATS")
)

# Forecast horizon
forecast_horizon = st.slider("Forecast Horizon (months)", min_value=12, max_value=36, value=24)

# Model implementation
if st.button("Run Forecast"):
    if model_option == "ARIMA":
        model = ARIMA()
    elif model_option == "ExponentialSmoothing":
        model = ExponentialSmoothing()
    elif model_option == "Linear Regression":
        model = LinearRegressionModel(lags=12)
    elif model_option == "TCN (Deep Learning)":
        model = TCNModel(input_chunk_length=24, output_chunk_length=12, n_epochs=10)
    elif model_option == "N-BEATS":
        model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=10)

    model.fit(scaled_series)
    forecast = model.predict(forecast_horizon)
    plot_series(forecast, title=f"{model_option} Forecast")

st.header("3. Univariate and Multivariate Time Series")
# Univariate Time Series
st.write("### Univariate Time Series")
plot_series(series, title="Univariate Time Series")

# Multivariate Time Series
st.write("### Multivariate Time Series (Sine Wave + Noisy Sine Wave)")
sine_series = sine_timeseries(length=100, value_frequency=0.05)
noisy_sine_series = sine_series + 0.2 * sine_timeseries(length=100, value_frequency=0.1)
multivariate_series = sine_series.stack(noisy_sine_series)
plot_series(multivariate_series, title="Multivariate Time Series")

st.header("4. Backtesting")
st.write("### Backtest ARIMA Model")
# Backtesting ARIMA
train, val = series.split_before(0.8)
model_arima = ARIMA()
model_arima.fit(train)
prediction_arima = model_arima.predict(len(val))

# Plot backtesting
fig, ax = plt.subplots(figsize=(10, 5))
train.plot(ax=ax, label="Train")
val.plot(ax=ax, label="Validation")
prediction_arima.plot(ax=ax, label="ARIMA Prediction")
ax.legend()
st.pyplot(fig)

st.write(f"MAPE (ARIMA): {mape(val, prediction_arima):.2f}%")

st.header("5. Ensembling Models")
# Updated ensemble using RegressionEnsembleModel
st.write("### NaiveSeasonal + NaiveDrift Ensembling")
naive_seasonal = NaiveSeasonal(K=12)
naive_drift = NaiveDrift()

ensemble = RegressionEnsembleModel(
    forecasting_models=[naive_seasonal, naive_drift],
    regression_train_n_points=24  # Number of points for training the regressor
)

ensemble.fit(series)
ensemble_forecast = ensemble.predict(forecast_horizon)
plot_series(ensemble_forecast, title="Regression Ensemble Model Forecast")

st.header("6. Handling Missing Data")
st.write("### Missing Data Example")
# Create a series with missing data
series_with_missing = series.copy()
series_with_missing = series_with_missing.drop_before(pd.Timestamp("1955-01"))

# Fill missing values
filled_series = fill_missing_values(series_with_missing, method="linear")
plot_series(filled_series, title="Filled Missing Data")

st.header("7. Probabilistic Forecasting (RNN)")
# Probabilistic Forecasting with RNN Model
st.write("### Probabilistic Forecasting with RNN Model (LSTM)")
model_rnn = RNNModel(
    model='LSTM',  # Type of RNN model
    input_chunk_length=12, 
    output_chunk_length=12, 
    n_epochs=10, 
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),  # Use Quantile Regression for probabilistic forecasting
    dropout=0.1
)

# Fit the model to the scaled series
model_rnn.fit(scaled_series)

# Predict future data with a probabilistic forecast
probabilistic_forecast = model_rnn.predict(forecast_horizon, num_samples=100)

# Plot the forecast with confidence intervals
fig, ax = plt.subplots(figsize=(10, 5))
probabilistic_forecast.plot(low_quantile=0.1, high_quantile=0.9, ax=ax)  # Show 80% confidence interval
ax.set_title("Probabilistic Forecast (RNN)")
st.pyplot(fig)




