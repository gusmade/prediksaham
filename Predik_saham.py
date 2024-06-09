import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime

# Fungsi untuk mengunduh dan memproses data saham
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    closing_prices = stock_data['Close'].values.reshape(-1, 1)
    return closing_prices

# Fungsi untuk menyiapkan data untuk LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Fungsi untuk membangun dan melatih model LSTM
def train_lstm_model(X_train, y_train, X_test, y_test, time_steps):
    model = Sequential()
    model.add(LSTM(units=64, activation='relu', input_shape=(time_steps, 1)))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    return model, history

# Fungsi untuk memprediksi harga saham
def predict_future_prices(model, last_30_days, scaler, time_steps, prediction_days):
    predictions = []
    scaled_last_30_days = scaler.transform(last_30_days)
    for _ in range(prediction_days):
        X_input = scaled_last_30_days.reshape((1, time_steps, 1))
        prediction = model.predict(X_input)
        predictions.append(prediction[0, 0])
        scaled_last_30_days = np.append(scaled_last_30_days[1:], prediction, axis=0)
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices

# Fungsi untuk memberikan saran berdasarkan prediksi harga
def provide_advice(predicted_prices):
    if predicted_prices[-1] > predicted_prices[0]:
        return "Saran: Pertahankan atau beli saham ini karena harga diprediksi akan naik."
    elif predicted_prices[-1] < predicted_prices[0]:
        return "Saran: Pertimbangkan untuk menjual saham ini karena harga diprediksi akan turun."
    else:
        return "Saran: Pertahankan saham ini karena harga diprediksi stabil."

# Aplikasi Streamlit
st.title('📈 Prediksi Harga Saham menggunakan LSTM 📉')

# Input untuk ticker saham
ticker = st.text_input('Masukkan Ticker Saham (misal: TLKM.JK):', 'TLKM.JK')

# Input untuk rentang tanggal
start_date = st.date_input('Tanggal Mulai', value=datetime(2020, 12, 22))
end_date = st.date_input('Tanggal Selesai', value=datetime(2024, 3, 15))

# Input untuk jumlah hari prediksi
prediction_days = st.number_input('Masukkan jumlah hari prediksi:', min_value=1, max_value=365, value=30)

# Tombol untuk mengambil dan memproses data saham
if st.button('Ambil Data Saham'):
    # Mengunduh dan memproses data saham
    closing_prices = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_closing_prices = scaler.fit_transform(closing_prices)

    # Membagi data menjadi set pelatihan dan pengujian
    train_size = int(len(scaled_closing_prices) * 0.8)
    train_data = scaled_closing_prices[:train_size]
    test_data = scaled_closing_prices[train_size:]

    # Menyiapkan data untuk LSTM
    time_steps = 30
    X_train, y_train = prepare_data(train_data, time_steps)
    X_test, y_test = prepare_data(test_data, time_steps)

    # Melatih model LSTM
    model, history = train_lstm_model(X_train, y_train, X_test, y_test, time_steps)

    # Evaluasi model
    loss, mae = model.evaluate(X_test, y_test)
    st.write(f"📉 **Loss:** {loss:.4f}, **Mean Absolute Error:** {mae:.4f}")

    # Memprediksi harga saham di masa depan
    last_30_days = closing_prices[-time_steps:]
    predicted_prices = predict_future_prices(model, last_30_days, scaler, time_steps, prediction_days)

    # Menampilkan harga yang diprediksi
    st.write(f'📅 **Prediksi Harga Saham untuk {prediction_days} Hari ke Depan:**')
    predicted_prices_df = pd.DataFrame(predicted_prices, columns=['Harga Prediksi'], index=pd.date_range(start=end_date, periods=prediction_days))
    st.dataframe(predicted_prices_df)

    # Plot harga yang diprediksi
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predicted_prices_df.index, predicted_prices_df['Harga Prediksi'], marker='o', linestyle='-', color='b')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Prediksi')
    ax.set_title(f'📈 Prediksi Harga Saham untuk {prediction_days} Hari ke Depan')
    st.pyplot(fig)

    # Memberikan saran kepada pengguna
    advice = provide_advice(predicted_prices)
    st.subheader("📊 Saran Investasi")
    st.write(advice)

# Form input di sidebar untuk pencarian user
st.sidebar.title("🔎 Pencarian Saham")
ticker_input = st.sidebar.text_input("Masukkan ticker saham:", "AAPL")
search_start_date = st.sidebar.date_input('Tanggal Mulai Pencarian', value=datetime(2020, 12, 22))
search_end_date = st.sidebar.date_input('Tanggal Selesai Pencarian', value=datetime(2024, 3, 15))
search_button = st.sidebar.button("Cari")

# Mengambil dan menampilkan data saham jika tombol pencarian diklik
if search_button:
    closing_prices_input = get_stock_data(ticker_input, search_start_date.strftime('%Y-%m-%d'), search_end_date.strftime('%Y-%m-%d'))
    st.sidebar.write(f"Harga penutupan untuk {ticker_input}:")
    st.sidebar.line_chart(closing_prices_input)

# Menjalankan aplikasi
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Muat ulang halaman untuk melihat prediksi terbaru atau mencari saham baru.")
