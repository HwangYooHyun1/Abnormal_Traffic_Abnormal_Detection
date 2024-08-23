#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, Dense
from datetime import timedelta
import matplotlib.pyplot as plt


# In[2]:


# 시간대별로 접근한 IP 주소의 개수를 구합니다.
ip_count_by_hour = df.groupby("datetime").nunique()

# 그래프를 그립니다.
plt.plot(ip_count_by_hour.index, ip_count_by_hour.values)
plt.title("Hourly Access IP Count")
plt.xlabel("Hour")
plt.ylabel("IP Count")
plt.show()


# In[49]:


data['datetime']


# In[6]:


# 로그 데이터에는 IP 주소와 datetime 정보가 포함되어 있습니다.
# IP 주소별로 접근한 datetime 정보를 추출합니다.
ip_datetime = data[['ip','datetime']]
ip_datetime.columns = ["ip", "datetime"]

# datetime 정보를 datetime 타입으로 변환합니다.
ip_datetime["datetime"] = pd.to_datetime(ip_datetime["datetime"])

# datetime 정보에서 시간대 정보를 추출합니다.
ip_datetime["hour"] = ip_datetime["datetime"].dt.hour

# IP 주소별로 시간대별 접근 수를 구합니다.
ip_hour_count = ip_datetime.groupby(["ip", "hour"]).count().reset_index()
ip_hour_count.columns = ["ip", "hour", "count"]

# 각 IP 주소별로 시간대별 접근 수를 그래프로 그립니다.
for ip in ip_hour_count["ip"].unique():
    ip_data = ip_hour_count[ip_hour_count["ip"] == ip]
    plt.plot(ip_data["hour"], ip_data["count"], "o-", label=ip)

# 그래프의 x 축과 y 축 레이블, 범례 등을 추가합니다.
plt.xlabel("Hour")
plt.ylabel("Access Count")
plt.title("Hourly Access Count of IPs")
#plt.legend(loc="upper left")

# 그래프를 출력합니다.
plt.show()


# In[13]:


# 접근 횟수가 높은 상위 5개 IP 추출
top_ips = data['ip'].value_counts().head(5).index.tolist()

# IP별 시간대별 접근 횟수 계산
ip_datetime = data[data['ip'].isin(top_ips)][['ip', 'datetime']]
ip_datetime['datetime'] = ip_datetime['datetime'].str[1:]
ip_datetime['datetime'] = pd.to_datetime(ip_datetime['datetime'], format="%d-%b %Y %H:%M:%S")
ip_datetime['hour'] = ip_datetime['datetime'].dt.hour
ip_hour_counts = ip_datetime.groupby(['ip', 'hour']).size().reset_index(name='count')

# 그래프 출력
fig, ax = plt.subplots(figsize=(12, 8))
for ip in top_ips:
    ip_data = ip_hour_counts[ip_hour_counts['ip'] == ip]
    ax.plot(ip_data['hour'], ip_data['count'], label=ip)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Access Count')
ax.set_title('Hourly Access Count for Top 5 IPs')
ax.legend()
plt.show()


# In[21]:


data[data['ip']=='180.178.99.174'].request.unique()


# In[2]:


access_log = pd.read_csv('AccessLogDataset.csv')


# In[4]:


access_log


# In[2]:


import numpy as np
import pandas as pd

# access log 파일을 읽어온다.
access_log = pd.read_csv('AccessLogDataset.csv')

# time 열을 datetime 형식으로 변환한다.
access_log['datetime'] = pd.to_datetime(access_log['datetime'], format="%d-%b %Y %H:%M:%S")

# 1초 간격으로 resample하고 count() 메서드를 이용해 각 구간별 로그 개수를 구한다.
log_count = access_log.resample('1S', on='datetime').count()['ip']

threshold = 30

# ddos 의심 구간 탐지
ddos_suspicion = log_count[log_count > threshold]

# ddos 의심 구간 중, ip별 로그 발생 횟수가 threshold를 넘는 경우 ddos 공격으로 판단
ddos_logs = []
for i in range(len(ddos_suspicion)):
    start_time = ddos_suspicion.index[i]
    end_time = ddos_suspicion.index[i] + pd.Timedelta(seconds=1)
    
    # 해당 시간대에 접속한 IP 주소를 추출하여 리스트로 저장
    ips = access_log.loc[(access_log['datetime'] >= start_time) & (access_log['datetime'] < end_time), 'ip'].tolist()
    
    # IP 주소별 로그 발생 횟수를 계산하여 ddos 공격으로 판단되는 경우 리스트에 추가
    for ip in set(ips):
        ip_count = ips.count(ip)
        if ip_count > threshold:
            ddos_logs.append((start_time, end_time, ip, ip_count))

# ddos로 판단된 구간의 시간, 해당 ip, 로그 발생 횟수 출력
for start_time, end_time, ip, count in ddos_logs:
    print(f"Time Range: {start_time} ~ {end_time}, IP: {ip}, Log Count: {count}")


# In[5]:


import numpy as np
import pandas as pd

# access log 파일을 읽어온다.
access_log = pd.read_csv('AccessLogDataset.csv')
access_log = access_log.dropna()

# time 열을 datetime 형식으로 변환한다.
access_log['datetime'] = pd.to_datetime(access_log['datetime'], format="%d-%b %Y %H:%M:%S")

# 5초 간격으로 resample하고 count() 메서드를 이용해 각 구간별 로그 개수를 구한다.
log_count = access_log.resample('1S', on='datetime').count()['ip']

threshold = 30

# ddos 의심 구간 탐지
ddos_suspicion = log_count[log_count > threshold]

# ddos_detected 열 추가
access_log['ddos_detected'] = False

# ddos 의심 구간 중, ip별 로그 발생 횟수가 threshold를 넘는 경우 ddos 공격으로 판단
ddos_logs = []
for i in range(len(ddos_suspicion)):
    start_time = ddos_suspicion.index[i]
    end_time = ddos_suspicion.index[i] + pd.Timedelta(seconds=1)
    
    # 해당 시간대에 접속한 IP 주소를 추출하여 리스트로 저장
    ips = access_log.loc[(access_log['datetime'] >= start_time) & (access_log['datetime'] < end_time), 'ip'].tolist()
    
    # IP 주소별 로그 발생 횟수를 계산하여 ddos 공격으로 판단되는 경우 리스트에 추가
    for ip in set(ips):
        ip_count = ips.count(ip)
        if ip_count > threshold:
            ddos_logs.append((start_time, end_time, ip, ip_count))

# ddos_detected 열 값 수정
for start_time, end_time, ip, count in ddos_logs:
    access_log.loc[(access_log['datetime'] >= start_time) & (access_log['datetime'] < end_time) & (access_log['ip'] == ip), 'ddos_detected'] = True

# ddos로 판단된 구간의 시간, 해당 ip, 로그 발생 횟수 출력
for start_time, end_time, ip, count in ddos_logs:
    print(f"Time Range: {start_time} ~ {end_time}, IP: {ip}, Log Count: {count}")


# In[6]:


access_log.isna().sum()


# In[7]:


access_log[access_log['ddos_detected']==True]
access_log.to_csv('ddos_result.csv', index=False)


# In[285]:


# ddos_logs에 저장된 데이터들 중에서 ip가 동일하고 시간대가 연속되는 경우를 묶어서 출력
ddos_groups = []
current_group = []
for i in range(len(ddos_logs)):
    if i == 0:
        current_group.append(ddos_logs[i])
    else:
        current_ip = current_group[0][2]
        current_end_time = current_group[-1][1]
        next_ip = ddos_logs[i][2]
        next_start_time = ddos_logs[i][0]
        if current_ip == next_ip and (next_start_time - current_end_time).seconds <= 5:
            current_group.append(ddos_logs[i])
        else:
            ddos_groups.append(current_group)
            current_group = [ddos_logs[i]]
    if i == len(ddos_logs) - 1:
        ddos_groups.append(current_group)

# 결과 출력
for group in ddos_groups:
    start_time = group[0][0]
    end_time = group[-1][1]
    ip = group[0][2]
    count = sum([x[3] for x in group])
    print(f"Time Range: {start_time} ~ {end_time}, IP: {ip}, Log Count: {count}")


# In[272]:


#ddos_logs dataframe 형태로 변환 
ddos_df= pd.DataFrame(ddos_logs, columns=['start_time', 'end_time', 'ip', 'count'])
print(ddos_df)


# In[295]:


for index, row in ddos_df.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']
    ip = row['ip']

    # 해당 구간에서 해당 IP의 referer 데이터를 추출
    referers = access_log[(access_log['ip'] == ip) & ((access_log['datetime'] == start_time) | (access_log['datetime'] == end_time))]['referer']
    
    # referer 데이터 출력
    print(f"IP {ip} in the time range {start_time} ~ {end_time} accessed the following referers:")
    print(referers.value_counts())
    print('\n')


# In[ ]:




