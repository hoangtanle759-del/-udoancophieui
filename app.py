import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------
# 1. Cấu hình trang Streamlit
# ----------------------------------------------------
st.set_page_config(
    page_title="Dự Báo Giá Cổ Phiếu bằng LSTM",
    page_icon="📈",
    layout="wide"
)

# Cố định Seed để ổn định kết quả
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu bằng Mô Hình LSTM")
st.markdown("Dự báo xu hướng giá cổ phiếu trong ngắn hạn và tính toán phần trăm **Lãi / Lỗ** kỳ vọng.")

# ----------------------------------------------------
# 2. Thanh công cụ bên trái (Sidebar Inputs)
# ----------------------------------------------------
st.sidebar.header("⚙️ Cấu Hình Dự Báo")

ticker = st.sidebar.text_input("Mã chứng khoán", value="VNM.VN", help="Đối với chứng khoán Việt Nam, thêm đuôi .VN (VD: VCB.VN, FPT.VN, HPG.VN)").upper()

start_date = st.sidebar.date_input("Ngày bắt đầu lấy dữ liệu", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("Ngày kết thúc", pd.to_datetime("2026-10-14"))

forecast_days = st.sidebar.slider("Số ngày muốn dự báo tương lai", min_value=1, max_value=10, value=10)

epochs = st.sidebar.slider("Số Epochs huấn luyện", min_value=5, max_value=30, value=10)

btn_run = st.sidebar.button("🚀 Bắt Đầu Dự Báo", type="primary")

# ----------------------------------------------------
# 3. Hàm xử lý & Huấn luyện
# ----------------------------------------------------
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

if btn_run or 'executed' in st.session_state:
    st.session_state['executed'] = True
    
    with st.spinner(f"Đang tải dữ liệu cổ phiếu {ticker}..."):
        df = yf.download(ticker, start=start_date, end=end_date)
        
    if df.empty or len(df) < 100:
        st.error(f"Không thể lấy đủ dữ liệu cho mã '{ticker}'. Vui lòng kiểm tra lại mã cổ phiếu hoặc khoảng thời gian!")
    else:
        # Xử lý trường hợp DataFrame trả về MultiIndex Columns của yfinance
        if isinstance(df.columns, pd.MultiIndex):
            close_data = df['Close'][ticker].values.reshape(-1, 1)
        else:
            close_data = df[['Close']].values
            
        st.success(f"Đã tải thành công **{len(close_data)}** phiên giao dịch của **{ticker}**!")
        
        # Split Data (80% Train, 20% Test)
        train_size = int(0.8 * len(close_data))
        train_data = close_data[:train_size]
        test_data = close_data[train_size:]
        
        # Scaling
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Sequence Data (Window Size = 60)
        window_size = 60
        if len(train_scaled) <= window_size or len(test_scaled) <= window_size:
            st.error("Dữ liệu quá ngắn để áp dụng Window Size = 60. Vui lòng chọn khoảng thời gian bắt đầu xa hơn!")
            st.stop()
            
        X_train, y_train = create_sequences(train_scaled, window_size)
        test_combined = np.vstack((train_scaled[-window_size:], test_scaled))
        X_test, y_test = create_sequences(test_combined, window_size)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Validation Set
        val_size = int(0.1 * len(X_train))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        
        # Build Model
        with st.spinner("Đang huấn luyện mô hình LSTM..."):
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )
            
        # Predict Test
        test_pred = model.predict(X_test)
        test_pred = scaler.inverse_transform(test_pred)
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
        mae = mean_absolute_error(y_test_real, test_pred)
        r2 = r2_score(y_test_real, test_pred)
        
        # Predict N Future Days
        last_60_days = np.vstack((train_scaled, test_scaled))[-window_size:]
        last_60_days = last_60_days.reshape(1, window_size, 1)
        
        future_predictions = []
        for _ in range(forecast_days):
            next_pred = model.predict(last_60_days, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_60_days = np.append(
                last_60_days[:, 1:, :],
                next_pred.reshape(1, 1, 1),
                axis=1
            )
            
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions)
        
        # ----------------------------------------------------
        # 4. Hiển thị kết quả & Đánh giá Lãi / Lỗ
        # ----------------------------------------------------
        last_real_price = float(close_data[-1][0])
        final_predicted_price = float(future_predictions[-1][0])
        price_change = final_predicted_price - last_real_price
        pct_change = (price_change / last_real_price) * 100
        
        st.subheader("🎯 Tóm Tắt Dự Báo Lãi / Lỗ")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá đóng cửa gần nhất", f"{last_real_price:,.2f}")
        col2.metric(f"Giá dự báo ngày thứ {forecast_days}", f"{final_predicted_price:,.2f}")
        col3.metric("Mức chênh lệch", f"{price_change:+,.2f}")
        col4.metric("Dự báo % Lãi / Lỗ", f"{pct_change:+.2f}%", delta=f"{pct_change:+.2f}%")
        
        st.markdown("---")
        
        # Bảng chi tiết dự báo từng ngày
        st.subheader(f"📋 Bảng Dự Báo Giá {forecast_days} Ngày Tới")
        
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        
        df_forecast = pd.DataFrame({
            "Ngày dự báo": future_dates.strftime('%d/%m/%Y'),
            "Giá dự báo": future_predictions.flatten(),
            "% Thay đổi so với giá gần nhất": [((p - last_real_price) / last_real_price) * 100 for p in future_predictions.flatten()]
        })
        
        df_forecast["Giá dự báo"] = df_forecast["Giá dự báo"].apply(lambda x: f"{x:,.2f}")
        df_forecast["% Thay đổi so với giá gần nhất"] = df_forecast["% Thay đổi so với giá gần nhất"].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(df_forecast, use_container_width=True)
        
        # ----------------------------------------------------
        # 5. Trực quan hóa dữ liệu (Biểu đồ)
        # ----------------------------------------------------
        st.subheader("📊 Biểu Đồ Trực Quan Hóa")
        
        tab1, tab2 = st.tabs(["Biểu đồ Tương tác (Plotly)", "Hàm Mất Mát (Train vs Val Loss)"])
        
        with tab1:
            fig = go.Figure()
            
            # Thực tế
            fig.add_trace(go.Scatter(
                y=y_test_real.flatten(),
                mode='lines',
                name='Giá thật (Test set)',
                line=dict(color='blue')
            ))
            
            # Dự báo Test
            fig.add_trace(go.Scatter(
                y=test_pred.flatten(),
                mode='lines',
                name='Dự báo (Test set)',
                line=dict(color='orange')
            ))
            
            # Dự báo tương lai
            future_x = list(range(len(y_test_real), len(y_test_real) + forecast_days))
            fig.add_trace(go.Scatter(
                x=future_x,
                y=future_predictions.flatten(),
                mode='lines+markers',
                name=f'Dự báo {forecast_days} ngày tới',
                line=dict(color='green', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title=f"So sánh Giá Thực tế vs Dự báo & {forecast_days} Ngày Tới của mã {ticker}",
                xaxis_title="Chỉ số phiên giao dịch",
                yaxis_title="Giá cổ phiếu",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig_loss, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history.history['loss'], label='Train Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title("Đồ thị Loss Train vs Validation")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss (MSE)")
            ax.legend()
            st.pyplot(fig_loss)
            
        # ----------------------------------------------------
        # 6. Đánh giá chỉ số mô hình
        # ----------------------------------------------------
        st.subheader("📐 Chỉ Số Đánh Giá Mô Hình (Model Metrics)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}")
        m_col2.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
        m_col3.metric("R² Score", f"{r2:.4f}")

else:
    st.info("👈 Vui lòng bấm nút **'🚀 Bắt Đầu Dự Báo'** ở thanh bên trái để khởi chạy mô hình!")
