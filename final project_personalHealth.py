#!/usr/bin/env python
# coding: utf-8

# In[187]:


import os
import flirt
import pandas as pd
import os
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from datetime import datetime, timedelta
import missingno as msno
import glob
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
import sympy as sy


# In[186]:


get_ipython().system('pip install sympy')


# In[74]:


def data_preprocess (sleepPath, rrPath, user):
    
    df = pd.read_csv(sleepPath, header = 0)
    new_df = df[['In Bed Date', 'In Bed Time', 'Out Bed Time', 'Total Minutes in Bed', 'Total Sleep Time (TST)', 'Number of Awakenings', 'Movement Index', 
            'Fragmentation Index']].copy()
    new_df['In Bed Time'], new_df['Out Bed Time'] = pd.to_datetime(new_df['In Bed Time'], format='%H:%M'), pd.to_datetime(new_df['Out Bed Time'], format='%H:%M')
    
    rr_1 = pd.read_csv(rrPath)
    rr_1 = rr_1[['day','time', 'ibi_s']]
    rr_1['time'] = pd.to_datetime(rr_1['time'], format = "%H:%M:%S")
    
    temp = pd.DataFrame(columns = rr_1.columns)
    temp1 = pd.DataFrame(columns = rr_1.columns)
    temp2 = pd.DataFrame(columns = rr_1.columns)

    for index, row in rr_1.iterrows():

            if new_df['In Bed Date'].min() == 2: 
                if row['time'] >= datetime.strptime(str(new_df['In Bed Time'].min().time()),'%H:%M:%S') and row['time'] <= datetime.strptime(str(new_df['Out Bed Time'].max().time()),'%H:%M:%S'):
                    temp = pd.concat([temp, pd.DataFrame([row])], axis=0, ignore_index=True)

            else:
                if row['time'] >= datetime.strptime(str(new_df['In Bed Time'].min().time()),'%H:%M:%S') and row['day'] == 1:
                    temp1 = pd.concat([temp1, pd.DataFrame([row])], axis=0, ignore_index=True)
                if row['time'] <= datetime.strptime(str(new_df['Out Bed Time'].max().time()),'%H:%M:%S') and row['day'] == 2:
                    temp2 = pd.concat([temp2, pd.DataFrame([row])], axis=0, ignore_index=True)


    if not temp1.empty:
        temp = pd.concat([temp1, temp2])

    temp['time'] = temp['time'].apply(lambda x: (datetime.strptime(str(x.time()),'%H:%M:%S') - datetime.strptime(str(temp[temp['day'] == new_df['In Bed Date'].min()]['time'][0].time()),'%H:%M:%S')))
    temp['time'] = temp['time'] - pd.to_timedelta(temp['time'].dt.days, unit='d')
    temp['time'] = temp['time'].apply(lambda x: x.total_seconds())
    temp.drop('day', axis=1, inplace=True)
    temp.rename(columns={'time': '1495624128.000000','ibi_s': 'ibi'}, inplace=True) 
    

    txt = "temp{userID}.csv"
    temp.to_csv(txt.format(userID=user), index=False)
    #path = "/Users/anushkahegde/Desktop/NEU/HINF_5300/temp{userID}.csv".format(userID=user)

    
    return temp


# In[76]:


f1_temp = data_preprocess("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_1/sleep.csv",
"/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_1/RR.csv", user=1)


# In[75]:


f2_temp = data_preprocess("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_2/sleep.csv",
"/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_2/RR.csv", user=2)


# In[77]:


f3_temp = data_preprocess("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_3/sleep.csv",
"/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_3/RR.csv", user=3)


# In[78]:


f4_temp = data_preprocess("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_4/sleep.csv",
"/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_4/RR.csv", user=4)


# In[79]:


f5_temp = data_preprocess("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_5/sleep.csv",
"/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_5/RR.csv", user=5)


# In[81]:


# #def check_file_path():
# #print( "Get cwd", os.getcwd()) 

# f3_temp[0].to_csv(os.path.join(path,'temp{user}.csv'.format(user=3)))


# In[68]:


# f1_temp = data_preprocess("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_1/sleep.csv",
# "/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_1/RR.csv", user=1)


# In[83]:


f1_hrv_features = get_testing_features('temp1.csv',"U1")
f1_hrv_features


# In[84]:


f2_hrv_features = get_testing_features('temp2.csv',"U2")
f3_hrv_features = get_testing_features('temp3.csv',"U3")
f4_hrv_features = get_testing_features('temp4.csv',"U4")
f5_hrv_features = get_testing_features('temp5.csv',"U5")

features = pd.concat([f1_hrv_features, f2_hrv_features, f3_hrv_features, f4_hrv_features, f5_hrv_features])
features


# In[80]:


def get_testing_features(csvFile, user):
    
    '''
    sleepPath
    rrPath
    user [int]
    
    '''
    
    
    ibis = flirt.reader.empatica.read_ibi_file_into_df(csvFile)

    hrv_features = flirt.get_hrv_features(ibis['ibi'], 60, 10, ["td", "fd", "stat", "nl"])
    hrv_features['User'] = str(user)
    
    return hrv_features
    


# In[146]:


# df = pd.DataFrame()
# df_list = []
# for u in range(len(user))
#     df = get_testing_features("temp{user}.csv".format(user=u),u)
#     df_list.append(df)

f1_hrv = get_testing_features("temp1.csv", "U1")
    


# In[160]:


f2_hrv_test = get_testing_features("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_2/sleep.csv",
"/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_2/RR.csv", user=2)


# In[170]:


f4_hrv_test = get_testing_features("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_4/sleep.csv",
"/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_4/RR.csv", user=4)


# In[162]:


files_sleep = sorted(glob.glob("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_*/sleep.csv"))

files_rr = sorted(glob.glob("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_*/RR.csv"))

user = []

sub = []
for f in files_sleep:
    res = re.findall("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_(\d+)/sleep.csv", f)
    sub.append(float(res[0]))
    
sub
    


# In[13]:


new_df = df[['Total Minutes in Bed', 'Total Sleep Time (TST)', 'Number of Awakenings', 'Movement Index', 
            'Fragmentation Index']].copy()


# In[34]:


df = pd.read_csv("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_3/sleep.csv", header = 0)
df


# In[35]:


new_df = df[['In Bed Date', 'In Bed Time', 'Out Bed Date', 'Out Bed Time', 'Total Minutes in Bed', 'Total Sleep Time (TST)', 'Number of Awakenings', 'Movement Index', 
            'Fragmentation Index']].copy()
new_df['In Bed Time'], new_df['Out Bed Time'] = pd.to_datetime(new_df['In Bed Time'], format='%H:%M'), pd.to_datetime(new_df['Out Bed Time'], format='%H:%M')
new_df['Out Bed Date'].replace([new_df['Out Bed Date'].max()], 2, inplace=True)
new_df


# In[36]:


import flirt.reader.empatica

rr_1 = pd.read_csv("/Users/anushkahegde/Desktop/NEU/HINF_5300/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/DataPaper/user_3/RR.csv")
rr_1 = rr_1[['day','time', 'ibi_s']]
rr_1['time'] = pd.to_datetime(rr_1['time'], format = "%H:%M:%S")
rr_1


# In[6]:


path = "/Users/anushkahegde/Desktop/NEU/HINF_5300/temp{userID}.csv".format(userID=1)
new_df['In Bed Time'].min().time()


# In[167]:


new_df.columns


# In[8]:


new_df['Out Bed Time'].max().time()


# In[53]:


temp = pd.DataFrame(columns = rr_1.columns)
temp1 = pd.DataFrame(columns = rr_1.columns)
temp2 = pd.DataFrame(columns = rr_1.columns)

for index, row in rr_1.iterrows():
    
        if new_df['In Bed Date'].min() == 2: 
            if row['time'] >= datetime.strptime(str(new_df['In Bed Time'].min().time()),'%H:%M:%S') and row['time'] <= datetime.strptime(str(new_df['Out Bed Time'].max().time()),'%H:%M:%S'):
                temp = pd.concat([temp, pd.DataFrame([row])], axis=0, ignore_index=True)
                
        else:
            if row['time'] >= datetime.strptime(str(new_df['In Bed Time'].min().time()),'%H:%M:%S') and row['day'] == 1:
                temp1 = pd.concat([temp1, pd.DataFrame([row])], axis=0, ignore_index=True)
            if row['time'] <= datetime.strptime(str(new_df['Out Bed Time'].max().time()),'%H:%M:%S') and row['day'] == 2:
                temp2 = pd.concat([temp2, pd.DataFrame([row])], axis=0, ignore_index=True)
            
            
if not temp1.empty:
    temp = pd.concat([temp1, temp2])
            

temp


# In[45]:


temp.reset_index().set_index("index", inplace=True)


# In[50]:


temp[temp['day'] == new_df['In Bed Date'].min()]['time'][0].time()


# In[54]:


temp['time'] = temp['time'].apply(lambda x: (datetime.strptime(str(x.time()),'%H:%M:%S') - datetime.strptime(str(temp[temp['day'] == new_df['In Bed Date'].min()]['time'][0].time()),'%H:%M:%S')))


# In[55]:


temp['time'] = temp['time'] - pd.to_timedelta(temp['time'].dt.days, unit='d')
temp


# In[56]:


temp['time'] = temp['time'].apply(lambda x: x.total_seconds())
temp


# In[113]:


temp['1495624128.000000'].value_counts()


# In[109]:


plt.figure(figsize=(10, 4))
plt.plot(temp['1495624128.000000'], temp['ibi'])
plt.title("RR interval P1")
plt.axis([0, 500, None, None])
plt.tight_layout()
plt.show()


# In[81]:


temp.rename(columns={'time': '1495624128.000000','ibi_s': 'ibi'}, inplace=True) #


# In[173]:


rr_1[rr_1['time'] >= datetime.strptime(str(new_df['Out Bed Time'].max().time()),'%H:%M:%S')]


# In[83]:


temp.to_csv("temp1.csv", index=False)


# In[120]:


txt = "temp{user}.csv"
# print(txt.format(price = 49))
path = temp.to_csv(txt.format(user=2))
path


# In[144]:


user = 1
str(user)


# In[121]:


path=r"/Users/anushkahegde/Desktop/NEU/HINF_5300/"
txt = "temp{userID}.csv"
temp.to_csv(path+txt.format(userID=3))

ibis1 = flirt.reader.empatica.read_ibi_file_into_df("/Users/anushkahegde/Desktop/NEU/HINF_5300/temp1.csv")


# In[114]:


import flirt.reader.empatica

ibis = flirt.reader.empatica.read_ibi_file_into_df("/Users/anushkahegde/Desktop/NEU/HINF_5300/temp1.csv")

hrv_features = flirt.get_hrv_features(ibis['ibi'], 60, 10, ["td", "fd", "stat", "nl"])


# In[108]:


hrv_features


# In[93]:


def get_features(featurefilepath, questfilepath, user):
    
    '''
    features [dataframe] takes the HRV and EDA features with a window size of 60 seconds and step size of 10 seconds
    for each participant
    quest [dataframe] contains the start and stop times for baseline and stress events
    the start and stop times are used to filter the feature_set [dataframe] for the required labels and values '''
    
    input_file = flirt.reader.empatica.read_ibi_file_into_df(featurefilepath)
    features = flirt.get_hrv_features(input_file['ibi'], 60, 10, ["td", "fd", "stat", "nl"])
    features.reset_index(inplace=True)
    features = features.rename(columns = {'datetime':'timestamp'})
    df = features.copy()
    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(str(x.time()),'%H:%M:%S') - datetime.strptime(str(df['timestamp'][0].time()),'%H:%M:%S'))
    
    quest = pd.read_csv(questfilepath, sep=';', header=1)
    quest = quest.iloc[0:2,0:3]
    base_start, base_stop, tsst_start, tsst_stop = quest.iloc[0,1], quest.iloc[1,1], quest.iloc[0,2], quest.iloc[1,2]


    feature_set = pd.DataFrame(columns=df.columns)
    
    for index, row in df.iterrows():
        
        # append baseline features
        if row['timestamp'] >= timedelta(seconds=base_start*60) and row['timestamp'] <= timedelta(seconds=base_stop*60):
            feature_set = pd.concat([feature_set, pd.DataFrame([row])], axis=0, ignore_index=True)

        # append stress features
        if row['timestamp'] >= timedelta(seconds=tsst_start*60) and row['timestamp'] <= timedelta(seconds=tsst_stop*60):
            feature_set = pd.concat([feature_set, pd.DataFrame([row])], axis=0, ignore_index=True)

    # add label 
    feature_set = feature_set.assign(label=[0 if x <= timedelta(seconds=base_stop*60) else 1 for x in feature_set['timestamp']])
    # add column for user
    feature_set['User'] = user
    
    return feature_set


# In[94]:


f1 = get_features("WESAD/S2/S2_E4_Data/IBI.csv", "WESAD/S2/S2_quest.csv", "U1")
f2 = get_features("WESAD/S3/S3_E4_Data/IBI.csv", "WESAD/S3/S3_quest.csv", "U2") 
f3 = get_features("WESAD/S4/S4_E4_Data/IBI.csv", "WESAD/S4/S4_quest.csv", "U3" )
f4 = get_features("WESAD/S5/S5_E4_Data/IBI.csv", "WESAD/S5/S5_quest.csv", "U4" )
f5 = get_features("WESAD/S6/S6_E4_Data/IBI.csv", "WESAD/S6/S6_quest.csv", "U5" )
f6 = get_features("WESAD/S7/S7_E4_Data/IBI.csv", "WESAD/S7/S7_quest.csv", "U6" )
f7 = get_features("WESAD/S8/S8_E4_Data/IBI.csv", "WESAD/S8/S8_quest.csv", "U7" )
f8 = get_features("WESAD/S9/S9_E4_Data/IBI.csv", "WESAD/S9/S9_quest.csv", "U8" )
f9 = get_features("WESAD/S10/S10_E4_Data/IBI.csv", "WESAD/S10/S10_quest.csv", "U9" )
f10 = get_features("WESAD/S11/S11_E4_Data/IBI.csv", "WESAD/S11/S11_quest.csv", "U10" )
f11 = get_features("WESAD/S13/S13_E4_Data/IBI.csv", "WESAD/S13/S13_quest.csv", "U11" )
f12 = get_features("WESAD/S14/S14_E4_Data/IBI.csv", "WESAD/S14/S14_quest.csv", "U12" )
f13 = get_features("WESAD/S15/S15_E4_Data/IBI.csv", "WESAD/S15/S15_quest.csv", "U13" )
f14 = get_features("WESAD/S16/S16_E4_Data/IBI.csv", "WESAD/S16/S16_quest.csv", "U14" )
f15 = get_features("WESAD/S17/S17_E4_Data/IBI.csv", "WESAD/S17/S17_quest.csv", "U15" )


# In[85]:


msno.matrix(features)


# In[ ]:


df['datetime'] = df['datetime'].apply(lambda x: datetime.strptime(str(x.time()),'%H:%M:%S') - datetime.strptime(str(df['datetime'][0].time()),'%H:%M:%S'))
df


# In[87]:


features.groupby('User').count()


# In[89]:


features.groupby('User').count().rsub(features.groupby('User').size(), axis=0)


# In[90]:


features.fillna(0)
features


# In[91]:





# In[107]:


training_features = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15])
training_features


# In[96]:


msno.matrix(training_features)


# In[100]:


training_features.groupby(['User', 'label']).count()


# In[102]:


training_features.describe()


# In[119]:


training_features[~training_features.isin([np.nan])]


# In[123]:


training_features[training_features.isna().any(axis=1)].groupby(['User','label']).count()


# In[112]:


training_features.drop('timestamp', axis=1).fillna(0)
training_features


# In[129]:


round(training_features.shape[1]/2)


# In[130]:


training_features = training_features.dropna(axis=0, thresh=round(training_features.shape[1]/2))
training_features


# In[131]:


training_features.groupby(['User','label']).count()


# In[134]:


model2 = LogisticRegression(random_state=1, solver='liblinear')


# In[135]:


X, y = training_features.drop(['label','User','timestamp'], axis=1).to_numpy(), training_features['label'].to_numpy()
groups = training_features['User'].to_numpy()
cv = LeaveOneGroupOut()


# In[138]:


scaler = StandardScaler() # standardized based on assumptions of data distribution for logistic regression
log_df_X = scaler.fit_transform(X)


# In[139]:


def get_scores():
    
    for s in ['accuracy', 'roc_auc', 'f1','precision','recall']:
        scores = cross_val_score(model2, log_df_X, y, scoring=s, cv=cv, n_jobs=-1, groups=groups)
        print(s, 'score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        
get_scores()


# In[147]:


features[features.isna().any(axis=1)].groupby('User').count()


# In[148]:


features.groupby("User").count()


# In[150]:


features = features.dropna(axis=0, thresh=round(features.shape[1]/2))
features.groupby("User").count()


# In[181]:


features


# ## Testing

# In[140]:


model2.fit(log_df_X, y)


# In[143]:


features.reset_index(inplace=True)


# In[151]:


test_X = features.drop(['User','datetime'], axis=1).to_numpy()
log_test_X = scaler.fit_transform(test_X)
test_groups = features['User'].to_numpy()


# In[153]:


array_prob = model2.predict_proba(log_test_X)


# In[161]:


array_prob[array_prob[:, 1] > 0.5]


# In[160]:


len(array_prob)


# In[162]:


def integrate(y_vals, h):
    i = 1
    total = y_vals[0] + y_vals[-1]
    for y in y_vals[1:-1]:
        if i % 2 == 0:
            total += 2 * y
        else:
            total += 4 * y
        i += 1
    return total * (h / 3.0)


# In[183]:


y_vals_1 = array_prob[:,1][-2431:]


# In[180]:


integrate(y_vals_1, 1)


# In[173]:


y_vals_2 = array_prob[:,1][2047:3362]
integrate(y_vals_2, 1)


# In[179]:


y_vals_01 = array_prob[:,0][-2431:]
integrate(y_vals_01, 1)


# In[177]:


integrate(y_vals_1, 1)/ (integrate(y_vals_1, 1)+ integrate(y_vals_01, 1))


# In[184]:


plt.figure(figsize=(10, 4))
plt.plot([i for i in range(0,len(y_vals_1))], y_vals_1)
plt.title("PDF")
#plt.axis([0, 500, None, None])
plt.tight_layout()
plt.show()


# In[206]:


x = [i for i in range(0,len(y_vals_1))]
plt.plot(x, y_vals_1)
plt.axhline(color='black')
plt.fill_between(x, f(x), where = [(x>0) and (x<2431) for x in x])


# In[207]:


x = sy.Symbol('x')
sy.integrate(f(x), (x, 0, 2430))


# In[201]:


def f(x): 
    return y_vals_1[x]


# In[204]:


y_vals_1[0]


# In[ ]:




