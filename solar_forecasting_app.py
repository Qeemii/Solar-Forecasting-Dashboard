import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go

# === Hyperparameter configs ===
MODEL_CONFIGS = {
    'gru': {'epochs': 32, 'batch_size': 16, 'learning_rate': 0.001},
    'lstm': {'epochs': 16, 'batch_size': 32, 'learning_rate': 0.01},
    'hybrid': {'epochs': 32, 'batch_size': 16, 'learning_rate': 0.001}
}

def load_data(uploaded_file, window_size=12, target_col='Plant Power(kW)_smoothed'):
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    # === PREPROCESSING ===
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    st.info(f"🧹 Removed {before - after} duplicate rows.")

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.set_index('date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df['Plant Power(kW)_smoothed'] = df['Plant Power(kW)'].ewm(span=12, adjust=False).mean()

    # === SCALING ===
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[df.columns] = scaler.fit_transform(df[df.columns])

    def create_sequences(df, target_col, window_size):
        X, y, dates = [], [], []
        target_idx = df.columns.get_loc(target_col)
        for i in range(window_size, len(df)):
            X.append(df.iloc[i - window_size:i].values)
            y.append(df.iloc[i, target_idx])
            dates.append(df.index[i])
        return np.array(X), np.array(y), np.array(dates)

    X, y, seq_dates = create_sequences(df_scaled, target_col, window_size)
    target_idx = df.columns.get_loc(target_col)

    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    date_test = seq_dates[train_size:]

    return X_train, y_train, X_test, y_test, date_test, scaler, target_idx, df.shape[1]

def inverse_single_column(scaled_column, target_idx, total_cols, scaler):
    dummy = np.zeros((len(scaled_column), total_cols))
    dummy[:, target_idx] = scaled_column
    return scaler.inverse_transform(dummy)[:, target_idx]

def build_model(model_type, input_shape, learning_rate):
    model = Sequential()
    if model_type == 'gru':
        model.add(GRU(64, return_sequences=False, input_shape=input_shape))
    elif model_type == 'lstm':
        model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    elif model_type == 'hybrid':
        model.add(GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(64, return_sequences=False))
    else:
        raise ValueError("Unknown model type.")
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    return model

def train_and_predict(model_type, X_train, y_train, X_test, scaler, target_idx, total_cols):
    config = MODEL_CONFIGS[model_type]
    model = build_model(model_type, X_train.shape[1:], config['learning_rate'])
    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], verbose=1)
    y_pred = model.predict(X_test).flatten()
    return inverse_single_column(y_pred, target_idx, total_cols, scaler), y_pred, model

def calculate_metrics(y_true_actual, y_pred_actual, y_pred_scaled, y_true_scaled):
    nmae = mean_absolute_error(y_true_scaled, y_pred_scaled)
    nrmse = np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    mape = np.mean(np.abs((y_true_actual - y_pred_actual) / y_true_actual)) * 100
    mase = mae / np.mean(np.abs(np.diff(y_true_actual)))
    r2 = r2_score(y_true_actual, y_pred_actual)
    return mae, rmse, mape, mase, r2, nmae, nrmse

def forecast_future(model, last_input, steps_ahead, target_idx, total_cols, scaler, last_date):
    forecasted_scaled = []
    current_input = last_input.copy()  # shape: (window_size, num_features)
    current_date = last_date

    for _ in range(steps_ahead):
        pred_scaled = model.predict(current_input[np.newaxis, :, :])[0, 0]
        forecasted_scaled.append(pred_scaled)

        # Build next time step row
        new_row = current_input[-1].copy()
        new_row[target_idx] = pred_scaled

        # Update time-based features for next day
        current_date += pd.Timedelta(days=1)
        new_row[total_cols - 3] = np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365.25)  # dayofyear_sin
        new_row[total_cols - 2] = np.cos(2 * np.pi * current_date.timetuple().tm_yday / 365.25)  # dayofyear_cos
        new_row[total_cols - 1] = current_date.month / 12.0                                       # month (normalized)

        # Slide window
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1] = new_row

    return inverse_single_column(np.array(forecasted_scaled), target_idx, total_cols, scaler)



# === Streamlit App ===
st.set_page_config(page_title="Train & Visualize Solar Forecast", layout="wide")
st.title("🔆 Solar Power Forecasting (GRU, LSTM, Hybrid GRU-LSTM)")

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
uploaded_report = st.file_uploader("📤 Upload Forecast Report (Optional)", type=["csv"], key="report")

# Initialize session state
if 'df_result' not in st.session_state:
    st.session_state.df_result = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None

# === Option 1: Load Forecast Report ===
if uploaded_report is not None:
    uploaded_report.seek(0)
    df_result = pd.read_csv(uploaded_report)
    required_cols = {'date', 'Actual', 'GRU', 'LSTM', 'Hybrid'}
    if not required_cols.issubset(df_result.columns):
        st.error("⚠️ Uploaded forecast report is missing required columns.")
        st.stop()
    df_result['date'] = pd.to_datetime(df_result['date'])
    st.session_state.df_result = df_result
    st.session_state.scaled_data = None
    st.success("✅ Forecast report loaded. No retraining needed.")

# === Option 2: Train New Models ===
if uploaded_file is not None and st.session_state.get('df_result') is None:
    uploaded_file.seek(0)
    df_preview = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df_preview.head(50))

    if st.button("🚀 Train All Models"):
        X_train, y_train, X_test, y_test_scaled, date_test, scaler, target_idx, total_cols = load_data(uploaded_file)
        st.session_state.X_test = X_test
        st.session_state.target_idx = target_idx
        st.session_state.total_cols = total_cols
        st.session_state.scaler = scaler

        y_test_actual = inverse_single_column(y_test_scaled, target_idx, total_cols, scaler)

        st.info("Training GRU...")
        y_gru_actual, y_gru_scaled, gru_model = train_and_predict('gru', X_train, y_train, X_test, scaler, target_idx, total_cols)
        st.session_state.gru_model = gru_model
        st.success("✅ GRU Done.")

        st.info("Training LSTM...")
        y_lstm_actual, y_lstm_scaled, lstm_model = train_and_predict('lstm', X_train, y_train, X_test, scaler, target_idx, total_cols)
        st.session_state.lstm_model = lstm_model
        st.success("✅ LSTM Done.")

        st.info("Training Hybrid GRU-LSTM...")
        y_hybrid_actual, y_hybrid_scaled, hybrid_model = train_and_predict('hybrid', X_train, y_train, X_test, scaler, target_idx, total_cols)
        st.session_state.hybrid_model = hybrid_model  # Save the model in session for later reuse
        st.success("✅ Hybrid GRU-LSTM Done.")

        df_result = pd.DataFrame({
            'date': date_test,
            'Actual': y_test_actual,
            'GRU': y_gru_actual,
            'LSTM': y_lstm_actual,
            'Hybrid': y_hybrid_actual
        })

        st.session_state.df_result = df_result
        st.session_state.scaled_data = {
            'GRU': y_gru_scaled,
            'LSTM': y_lstm_scaled,
            'Hybrid': y_hybrid_scaled,
            'Actual': y_test_scaled
        }

        # Allow forecast result download
        st.subheader("📥 Download Forecast Report")
        csv_result = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast Report (CSV)",
            data=csv_result,
            file_name='forecast_report.csv',
            mime='text/csv'
        )

# === Display Forecast & Metrics ===
if st.session_state.get('df_result') is not None:
    df_result = st.session_state.df_result

    st.subheader("📈 Forecast Comparison")
    show_gru = st.checkbox("Show GRU", value=False)
    show_lstm = st.checkbox("Show LSTM", value=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_result['date'], y=df_result['Actual'], name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_result['date'], y=df_result['Hybrid'], name='Hybrid GRU-LSTM', line=dict(color='green', dash='dot')))
    if show_gru:
        fig.add_trace(go.Scatter(x=df_result['date'], y=df_result['GRU'], name='GRU', line=dict(dash='dash')))
    if show_lstm:
        fig.add_trace(go.Scatter(x=df_result['date'], y=df_result['LSTM'], name='LSTM', line=dict(dash='dashdot')))
    fig.update_layout(title="Forecasted Plant Power Output", xaxis_title="Date", yaxis_title="Plant Power (kW)", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # === Metrics ===
    if st.session_state.get('scaled_data') is not None:
        st.subheader("📊 Evaluation Metrics")
        scaled_data = st.session_state.scaled_data
        metrics_data = []
        for name in ['GRU', 'LSTM', 'Hybrid']:
            mae, rmse, mape, mase, r2, nmae, nrmse = calculate_metrics(
                y_true_actual=df_result['Actual'].values,
                y_pred_actual=df_result[name].values,
                y_pred_scaled=scaled_data[name],
                y_true_scaled=scaled_data['Actual']
            )
            metrics_data.append([name, nmae, nrmse, mape, mase, r2])

        metrics_df = pd.DataFrame(metrics_data, columns=["Model", "MAE", "RMSE", "MAPE (%)", "MASE", "R²"])
        st.dataframe(metrics_df.set_index("Model").style.format("{:.4f}"))
    else:
        st.info("Only forecast comparison is available. Retrain the model to see evaluation metrics.")

else:
    st.info("Please upload your CSV dataset or load a forecast report to begin.")

# === Forecast next 1 year (8760 steps) ===
if all(m in st.session_state for m in ['gru_model', 'lstm_model', 'hybrid_model']):
    st.subheader("📅 Forecast Next Year (All Models)")

    if st.button("📈 Generate 1-Year Forecast (GRU, LSTM, Hybrid)"):
        steps_ahead = 365  # ✅ for daily data
        st.info("⏳ Forecasting 1 year ahead...")

        # Get last input sequence from test set
        last_input = st.session_state.X_test[-1]
        target_idx = st.session_state.target_idx
        total_cols = st.session_state.total_cols
        scaler = st.session_state.scaler


        future_gru = forecast_future(
            st.session_state.gru_model,
            last_input,
            steps_ahead,
            target_idx,
            total_cols,
            scaler,
            last_date=pd.to_datetime(df_result['date'].max())
        )

        future_lstm = forecast_future(
            st.session_state.lstm_model,
            last_input,
            steps_ahead,
            target_idx,
            total_cols,
            scaler,
            last_date=pd.to_datetime(df_result['date'].max())
        )

        future_hyrbid = forecast_future(
            st.session_state.hyrbid_model,
            last_input,
            steps_ahead,
            target_idx,
            total_cols,
            scaler,
            last_date=pd.to_datetime(df_result['date'].max())
        )

        last_date = pd.to_datetime(df_result['date'].max())
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps_ahead, freq='D')


        df_future = pd.DataFrame({
            'date': future_dates,
            'GRU_Forecast': future_gru,
            'LSTM_Forecast': future_lstm,
            'Hybrid_Forecast': future_hybrid
        })

        st.success("✅ All forecasts complete.")
        st.dataframe(df_future.head(48))

        st.download_button(
            label="⬇️ Download All 1-Year Forecasts (CSV)",
            data=df_future.to_csv(index=False).encode('utf-8'),
            file_name='future_forecast_all_models.csv',
            mime='text/csv'
        )

        # Plotting
        st.subheader("📊 Forecast Comparison Plot (Next 1 Year)")
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df_future['date'], y=df_future['GRU_Forecast'], name='GRU', line=dict(dash='dash')))
        fig_future.add_trace(go.Scatter(x=df_future['date'], y=df_future['LSTM_Forecast'], name='LSTM', line=dict(dash='dot')))
        fig_future.add_trace(go.Scatter(x=df_future['date'], y=df_future['Hybrid_Forecast'], name='Hybrid GRU-LSTM', line=dict(color='green')))
        fig_future.update_layout(title="1-Year Forecast: GRU vs LSTM vs Hybrid", xaxis_title="Date", yaxis_title="Forecast Power (kW)", hovermode="x unified", height=500)
        st.plotly_chart(fig_future, use_container_width=True)
else:
    st.warning("⚠️ Please train all models first before running 1-year forecast.")

