#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
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


# 정답(label)이 false인 데이터 개수를 세고, 절반을 삭제할 개수로 계산합니다.
false_count = len(df) - df['label'].sum()
delete_count = int(false_count * 0.5)

# 정답(label)이 false인 데이터 중에서 삭제할 개수만큼 무작위로 선택하여 삭제합니다.
delete_indices = df[df['label'] == False].sample(delete_count).index
df = df.drop(delete_indices)


# In[ ]:


df


# ## IP 전처리 

# In[4]:


cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))


# In[6]:


df


# ## One-Class VCM 

# In[7]:


# 데이터 준비
x = df.drop(['label'], axis=1).values
y = df['label'].values


# In[8]:


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


# In[9]:


# 모델 학습
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stopping])


# In[10]:


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


# In[11]:


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


# In[ ]:


from keras.models import save_model

# 모델 저장
save_model(model, 'access_model(2).h5')

