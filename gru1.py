# 출처: https://welldonecode.tistory.com/97

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import mse
from tensorflow.keras.metrics import RootMeanSquaredError, mean_squared_error, MeanAbsoluteError, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# TensorFlow 로그 메시지 숨기기
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 데이터 로드 및 전처리
agg_10min = pd.read_csv('aggregation_10min.csv', engine='python')
agg_10min = agg_10min.drop(['Unnamed: 0', 'timestamp', 'num_gpu'], axis=1)

# 데이터 스케일링 (Min-Max 스케일링)
scaler = MinMaxScaler()
agg_10min['gpu_milli'] = scaler.fit_transform(agg_10min['gpu_milli'].values.reshape(-1, 1))

# 슬라이딩 윈도우 설정
window_size = 3  # 윈도우 크기 (조정 가능)
step_size = 1    # 윈도우 이동 간격 (조정 가능)

# 슬라이딩 윈도우를 사용하여 데이터 생성
def create_sliding_window_data(data, lookback_time=5, predict_time=2):
    X = []
    y = []

    for i in range(len(data) - (lookback_time - 1) - predict_time):
        x_window = data['gpu_milli'].iloc[i:i+lookback_time].values
        y_value = data['gpu_milli'].iloc[i+lookback_time+predict_time-1]

        X.append(x_window)
        y.append(y_value)

    return np.array(X), np.array(y)

x, t = create_sliding_window_data(agg_10min)

# 데이터 분할 (학습 및 테스트 데이터)
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, shuffle=False)

# GRU 모델 생성
cell_size = 256
timesteps = 5
feature = 1

model = Sequential(name="GPU_GRU")
model.add(GRU(cell_size, input_shape=(timesteps, feature), return_sequences=True))
model.add(GRU(cell_size))
model.add(Dense(1))

model.compile(loss=mse, optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
model.summary()



# 모델 학습
history = model.fit(x_train, t_train, epochs=100, batch_size=64, validation_data=(x_test, t_test), verbose=1)


# 모델 평가
y_pred = model.predict(x_test)
t_test_reset = scaler.inverse_transform(t_test.reshape(-1, 1))  # t_test를 2D 배열로 변환
y_pred_reset = scaler.inverse_transform(y_pred)  # y_pred는 이미 2D 배열이므로 추가 변환이 필요 없음

r2_reset = r2_score(t_test_reset, y_pred_reset)
loss_reset = mean_squared_error(t_test_reset, y_pred_reset)
rmse_reset = np.sqrt(mean_squared_error(t_test_reset, y_pred_reset))
mae_reset = mean_absolute_error(t_test_reset, y_pred_reset)
mape_reset = np.mean(np.abs((t_test_reset - y_pred_reset) / t_test_reset)) * 100

r2_scale = r2_score(t_test, y_pred)
loss_scale = mean_squared_error(t_test, y_pred)
rmse_scale = np.sqrt(mean_squared_error(t_test, y_pred))
mae_scale = mean_absolute_error(t_test, y_pred)
mape_scale = np.mean(np.abs((t_test - y_pred) / t_test)) * 100

print('===Decode Data===')
print('Test Loss (MSE):', loss_reset)
print('Test RMSE:', rmse_reset)
print('Test MAE:', mae_reset)
print('R-squared (R^2):', r2_reset)
print('Test MAPE:', mape_reset)

print('===Scaled Data===')
print('Test Loss (MSE):', loss_scale)
print('Test RMSE:', rmse_scale)
print('Test MAE:', mae_scale)
print('R-squared (R^2):', r2_scale)
print('Test MAPE:', mape_scale)


# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(t_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('gpu_milli')
plt.title('gpu_milli Prediction')
plt.legend()
plt.show()