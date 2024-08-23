#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


# In[42]:


df = pd.read_csv('access_label.csv')


# In[43]:


# 시간대별로 IP를 그룹화하여 접근한 IP 개수를 구합니다.
ip_count_by_hour = df.groupby(["datetime", "IP"]).size().groupby("datetime").cumsum()


# In[44]:


# 정답(label)이 false인 데이터 개수를 세고, 절반을 삭제할 개수로 계산합니다.
false_count = len(df) - df['label'].sum()
delete_count = int(false_count * 0.5)

# 정답(label)이 false인 데이터 중에서 삭제할 개수만큼 무작위로 선택하여 삭제합니다.
delete_indices = df[df['label'] == False].sample(delete_count).index
df = df.drop(delete_indices)


# In[29]:


df


# In[33]:


df = df.drop('label', axis=1)


# ## 데이터 탐색

# In[6]:


#datetime 형식 변환 
df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
df.info()


# In[7]:


#시간 컬럼 생성 
df['hour'] = df['datetime'].dt.hour

df_line = df
df['access_count'] = df.groupby('datetime')['datetime'].transform('count')

df_line = df_line.groupby('datetime')['access_count'].sum().reset_index()


# In[8]:


# 시간별 이동 평균 생성 
df_line['hour'] = df_line['access_count'].rolling(window=15).mean()

# 선그래프 시각화 
ax=df_line.plot(x='datetime', y='access_count', linewidth='0.5')
df_line.plot(x='datetime',y='hour',color='#FF7F50',linewidth='1', ax=ax)


# In[30]:


cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))


# In[31]:


import seaborn as sns 

sns.set(font_scale=1.1)
sns.pairplot(df, diag_kind='kde')
plt.show()


# In[34]:


#피어슨 상관계수 산출
df.corr(method='pearson')
sns.clustermap(df.corr(), annot=True, cmap='RdYlBu_r',vmin=-1, vmax=1)


# ## IP 전처리 

# In[ ]:


df.info()


# In[45]:


cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))


# In[46]:


ms = MinMaxScaler()
df_ms = ms.fit_transform(df)

df_ms = pd.DataFrame(data=df_ms, columns=df.columns)

df_ms


# In[ ]:


from sklearn.decomposition import PCA
import seaborn as sns 


# In[ ]:


#주성분 개수 설정(최대 수 설정)
pca = PCA(n_components=8)
df_pca = pca.fit_transform(df_ms)

#주성분으로 변형된 테이블 생성
df_pca = pd.DataFrame(data=df_pca)

np.round_(pca.explained_variance_ratio_,3)


# In[ ]:


#주성분 개수 설정(2개 설정)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_ms)

df_pca = pd.DataFrame(data=df_pca, columns=["c1","c2"])
df_pca.head()


# ## 가중치 적용 

# In[47]:


# 가중치를 부여할 학습 데이터를 선택 (동일한 시간에 접근한 데이터의 개수에 따라 가중치 부여)
df['access_count'] = df.groupby('datetime')['datetime'].transform('count')

# 가중치를 부여 (조건: 동일한 시간에 접근한 데이터의 개수에 따라 가중치 부여)
df['weight'] = df['access_count'] / df['access_count'].max()


# In[48]:


sum(df['weight']==1)
df = df.drop('access_count', axis=1)


# In[49]:


df


# ## LSTM 

# In[50]:


# 데이터 준비
x = df.drop(['label'], axis=1).values
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


# In[53]:


model.summary()


# In[ ]:


# 모델 학습
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stopping])


# In[ ]:


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


# In[ ]:


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
save_model(model, 'access_model.h5')

