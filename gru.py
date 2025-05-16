# 출처: https://welldonecode.tistory.com/97

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import mse
from tensorflow.keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# TensorFlow 로그 메시지 숨기기
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 데이터 로드 및 전처리
closing_price = pd.read_csv('feature_drop.csv', engine='python')
closing_price = closing_price.drop(['Unnamed: 0', 'date'], axis=1)

# 데이터 스케일링 (Min-Max 스케일링)
scaler = MinMaxScaler()
closing_price['close'] = scaler.fit_transform(closing_price['close'].values.reshape(-1, 1))

# 슬라이딩 윈도우 설정
window_size = 3  # 윈도우 크기 (조정 가능)
step_size = 1    # 윈도우 이동 간격 (조정 가능)

# 슬라이딩 윈도우를 사용하여 데이터 생성
def create_sliding_window_data(data, lookback_time=5, predict_time=2):
    X = []
    y = []

    for i in range(len(data) - (lookback_time - 1) - predict_time):
        x_window = data['close'].iloc[i:i+lookback_time].values
        y_value = data['close'].iloc[i+lookback_time+predict_time-1]

        X.append(x_window)
        y.append(y_value)

    return np.array(X), np.array(y)

x, t = create_sliding_window_data(closing_price)

# 데이터 분할 (학습 및 테스트 데이터)
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, shuffle=False)

# GRU 모델 생성
cell_size = 256
timesteps = 5
feature = 1

model = Sequential(name="GPU_GRU")
model.add(GRU(cell_size, input_shape=(timesteps, feature), return_sequences=True))
model.add(GRU(cell_size))
model.add(Dense(1))

model.compile(loss=mse, optimizer='adam', metrics=['accuracy'])
model.summary()



# 모델 학습
start = datetime.datetime.now()
history = model.fit(x_train, t_train, epochs=400, batch_size=170, validation_data=(x_test, t_test), verbose=1)
end = datetime.datetime.now()


# 모델 평가
y_pred = model.predict(x_test)
t_test_reset = scaler.inverse_transform(t_test.reshape(-1, 1))  # t_test를 2D 배열로 변환
y_pred_reset = scaler.inverse_transform(y_pred)  # y_pred는 이미 2D 배열이므로 추가 변환이 필요 없음

m2 = MeanSquaredError()
m2.update_state(t_test_reset, y_pred_reset)
rmse_reset = np.sqrt(m2.result())
m3 = MeanAbsoluteError()
m3.update_state(t_test_reset, y_pred_reset)
mae_reset = m3.result()
mape_reset = np.mean(np.abs((t_test_reset - y_pred_reset) / t_test_reset)) * 100

m5 = MeanSquaredError()
m5.update_state(t_test, y_pred)
rmse_scale = np.sqrt(m5.result())
m6 = MeanAbsoluteError()
m6.update_state(t_test, y_pred)
mae_scale = m6.result()
mape_scale = np.mean(np.abs((t_test - y_pred) / t_test)) * 100

print('===Decode Data===')
print('Test RMSE:', rmse_reset)
print('Test MAPE:', mape_reset)

print('===Scaled Data===')
print('Test RMSE:', rmse_scale)
print('Test MAPE:', mape_scale)

time = end - start
print('학습시간: ', time)


# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(t_test, label='Actual', color='red')
plt.plot(y_pred, label='Predicted', color='blue', linestyle='--')
plt.xlabel('Day')
plt.ylabel('Closing Price')
plt.title('GRU Model Prediction')
plt.legend()
plt.show()


# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(t_test_reset, label='Actual', color='red')
plt.plot(y_pred_reset, label='Predicted', color='blue', linestyle='--')
plt.xlabel('Day')
plt.ylabel('Closing Price')
plt.title('GRU Model Prediction')
plt.legend()
plt.show()