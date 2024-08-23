#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from pyod.models.vae import VAE
from pyod.models.ocsvm import OCSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


# In[2]:


df = pd.read_csv('access_label.csv')


# In[3]:


# 시간대별로 IP를 그룹화하여 접근한 IP 개수를 구합니다.
ip_count_by_hour = df.groupby(["datetime", "IP"]).size().groupby("datetime").cumsum()


# In[ ]:


# 그룹별 누적 막대 그래프를 그립니다.
num_subplots = len(ip_count_by_hour.groupby("datetime"))  # 그룹화된 데이터의 개수
num_rows = (num_subplots + 1) // 2  # 서브플롯의 행 개수
fig, axs = plt.subplots(num_rows, 2, figsize=(12, num_rows * 4), sharex=True, sharey=True)  # 서브플롯 생성

# 서브플롯에 그래프 그리기
for i, (hour, count) in enumerate(ip_count_by_hour.groupby("datetime")):
    ax = axs[i // 2, i % 2]  # 서브플롯 위치 지정
    ax.bar(count.index.get_level_values(0), count.values, label=str(hour), alpha=0.5)
    ax.set_title("Hour: {}".format(hour))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cumulative IP Count")
    ax.legend()

# 그래프 간격 조정
plt.tight_layout()

# 출력
plt.show()


# In[3]:


# 정답(label)이 false인 데이터 개수를 세고, 절반을 삭제할 개수로 계산합니다.
false_count = len(df) - df['label'].sum()
delete_count = int(false_count * 0.5)

# 정답(label)이 false인 데이터 중에서 삭제할 개수만큼 무작위로 선택하여 삭제합니다.
delete_indices = df[df['label'] == False].sample(delete_count).index
df = df.drop(delete_indices)


# In[4]:


df.head()


# ## IP 전처리 

# In[5]:


df.info()


# In[6]:


cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# 데이터를 청크 단위로 로드하고 처리하기
chunk_size = 10000  # 각 청크의 크기를 설정합니다.
freq_table = pd.DataFrame()  # 빈 데이터프레임을 생성합니다.

# 청크 단위로 데이터를 로드하고 처리합니다.
for i in range(0, len(df), chunk_size):
    chunk = df.head(chunk_size)
    chunk_no_label = chunk.drop(columns=['label'])
    chunk_freq_table = chunk_no_label.groupby(["IP", "datetime"]).size().reset_index(name="count")
    freq_table = freq_table.append(chunk_freq_table, ignore_index=True)

# 중복 값을 집계합니다.
freq_table = freq_table.groupby(["IP", "datetime"]).size().reset_index(name="count")

# 피벗 테이블 생성
freq_table_chunk = freq_table.pivot_table(index="IP", columns="datetime", values="count", aggfunc="sum", fill_value=0)

# 히트맵 그리기
plt.figure(figsize=(12, 6))
sns.heatmap(freq_table_chunk, cmap="YlGnBu")
plt.show()


# ## 가중치 적용 

# In[7]:


# 가중치를 부여할 학습 데이터를 선택 (동일한 시간에 접근한 데이터의 개수에 따라 가중치 부여)
df['access_count'] = df.groupby('datetime')['datetime'].transform('count')

# 가중치를 부여 (조건: 동일한 시간에 접근한 데이터의 개수에 따라 가중치 부여)
df['weight'] = df['access_count'] / df['access_count'].max()


# In[8]:


sum(df['weight']==1)


# In[9]:


df


# ## LSTM 

# In[10]:


# 데이터 준비
x = df.drop(['label','access_count'], axis=1).values
y = df['label'].values
# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# 데이터를 train-validation-test 세트로 분할
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

# PCA를 사용하여 차원 축소
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# 데이터의 차원(특성 수) 확인
n_features = X_scaled.shape[1]

# LSTM 입력에 맞게 데이터 형태 변환
X_train = X_train.reshape((X_train.shape[0], 1, n_features))
X_val = X_val.reshape((X_val.shape[0], 1, n_features))
X_test = X_test.reshape((X_test.shape[0], 1, n_features))

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(1, n_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16, activation='relu'))
model.add(Dropout(0.2))  # 추가적인 Dropout 레이어
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# 조기 종료 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# In[ ]:


model


# In[11]:


# 모델 학습
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stopping])


# In[14]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 훈련 데이터로 예측
y_train_pred = model.predict(X_train)
y_train_pred = (y_train_pred > 0.5).astype(int)

# 훈련 세트의 평가 지표 계산
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# 테스트 데이터로 예측
y_test_pred = model.predict(X_test)
y_test_pred = (y_test_pred > 0.5).astype(int)

# 테스트 세트의 평가 지표 계산
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)


# In[15]:


# 평가 지표 출력
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print('\n')

print("Train Precision:", train_precision)
print("Test Precision:", test_precision)
print('\n')

print("Train Recall:", train_recall)
print("Test Recall:", test_recall)
print('\n')

print("Train F1 Score:", train_f1)
print("Test F1 Score:", test_f1)


# In[23]:


from keras.models import save_model

# 모델 저장
save_model(model, 'access_model.h5')

