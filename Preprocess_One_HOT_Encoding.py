#!/usr/bin/env python 
# coding: utf-8 
 
# # 1. Imports 
 
# In[1]: 
 
 
import pandas as pd import os import pickle import numpy as np from matplotlib import pyplot as plt from sklearn.utils import shuffle from sklearn.preprocessing import LabelBinarizer as lbe 
 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, 
MaxPooling1D, LSTM 
 
 
# # 2. Data Reading 
 
# In[2]: 
 
 
 
 
data_xls = pd.read_excel('assets/data/unmod/dataPhish/1.xlsx',engine='openpyxl',index_ col=None) 
data_xls1 = pd.read_excel('assets/data/unmod/dataPhish/2.xlsx',engine='openpyxl',index_ col=None) data_xls = data_xls.append(data_xls1, ignore_index = True) 
 
data_xls1 = pd.read_excel('assets/data/unmod/dataPhish/3.xlsx',engine='openpyxl',index_ col=None) data_xls = data_xls.append(data_xls1, ignore_index = True) 
 
data_xls1 = pd.read_excel('assets/data/unmod/dataPhish/4.xlsx',engine='openpyxl',index_ col=None) data_xls = data_xls.append(data_xls1, ignore_index = True) 
 
# data_xls = data_xls.sample(2200) 
 
 
data_xls 
 
 
# In[3]: 
 
 
 
path="assets/data/unmod/dataValid" data_csv = pd.read_csv(path+'/part1.csv',header=0) for i in range(2,5): 
    data_csv1 = pd.read_csv(path+'/part'+str(i)+'.csv',header=0)     data_csv = data_xls.append(data_csv1, ignore_index = True) data_csv = data_csv.sample(4649) data_csv 
 
 
# In[4]: 
 
 
dataPhish = data_xls[["Subject","Content"]] dataPhish['text'] = dataPhish['Subject'] +dataPhish['Content'] dataPhish = dataPhish[["text"]] 
 
dataValid = data_csv[["Subject","Content"]] dataValid['text'] = dataValid['Subject'] +dataValid['Content'] dataValid = dataValid[["text"]] 
 
dataPhish["label"] = 1 dataValid["label"] = 0 
 
 
# In[5]: 
 
 
dataPhish = dataPhish.dropna() dataValid = dataValid.dropna() 
 
dataValid['text'] = dataValid['text'].str.replace(r'\\[nt]+', ' ') dataPhish['text'] = dataPhish['text'].str.replace(r'\\[nt]+', ' ') 
 
 
# In[6]: 
 
 
dataPhish 
 
 
# In[7]: 
 
 
lenghtsValid = [] for i in dataValid[['text']].values:     lenghtsValid.append(len(str(i[0]))) 
 
     
def Average(lst): 
    return sum(lst) / len(lst) 
 
print(np.percentile(lenghtsValid, 90)) print(Average(lenghtsValid)) 
 
 
# In[8]: 
 
 
lenghtsPhish = [] 
for i in dataPhish[['text']].values:     lenghtsPhish.append(len(str(i[0]))) 
 
     
def Average(lst): 
    return sum(lst) / len(lst) 
 
print(np.percentile(lenghtsPhish, 90)) print(Average(lenghtsPhish)) 
 
 
# In[9]: 
 
 
combined = lenghtsValid+lenghtsPhish print("Total : ",len(combined)) combined.sort(reverse=True) print(np.percentile(combined, 90)) print(Average(combined)) print(max(combined)) 
 
 
# In[10]: 
 
 
plt.plot(lenghtsValid) plt.plot(lenghtsPhish) plt.show() 
 
 
# # 3. New one hot encoding 
 
# In[11]: 
 
 
df = pd.concat([dataValid, dataPhish], ignore_index=True) 
 
 
# In[19]: 
 
 
df = df.sample(frac=1).reset_index(drop=True) df 
 
 
# In[23]: 
 
 
label_counts = df['label'].value_counts() 
 
# Plot the label frequencies as a bar plot fig, ax = plt.subplots() label_counts.plot.bar(ax=ax) 
 
# Add the exact count of each label to the plot 
for i, count in enumerate(label_counts):     ax.text(i, count+1, str(count), ha='center') 
 
# Add axis labels and title to the plot plt.xlabel('Label') plt.ylabel('Frequency') plt.title('Label Frequencies') 
 
# Display the plot plt.show() 
 
 
# In[44]: 
 
 
num_files = 20  # The number of files to create batch_size = len(df) // num_files 
 
for i in range(num_files):     filename = f'assets/data/data_{i}.csv'     start = i * batch_size     end = (i + 1) * batch_size if i < num_files - 1 else len(df)     df_batch = df[start:end]     df_batch.to_csv(filename, index=False) 
 
 
# In[45]: 
 
 
 classes = ['\n','\t','\r'] start = 32 for i in range(start,start+95):     classes.append(chr(i)) print(len(classes)) 
 
lb=lbe() lb.classes_=np.array(list(classes)) 
 
with open('assets/label/labels', 'wb') as f: 
    pickle.dump(lb, f) 
 
 
# In[52]: 
 
 
filenames = [] 
 
for i in range(20): 
    filenames.append('assets/data/data_{}.csv'.format(i)) filenames 
 
 
# In[51]: 
 
 
 
lb = None 
# Open the pickle file in read binary mode with open('assets/label/labels', 'rb') as f: 
    # Load the contents of the file using pickle.load()     lb = pickle.load(f) 
 
# Use the loaded data lb 
 
 
# In[53]: 
 
 
max_len = 4000 num_classes = 98 result = [] for filename in filenames: 
    # Load the CSV file into a Pandas DataFrame     df = pd.read_csv(filename) 
    # Create a dataset from the DataFrame and apply the parse_data() function     text = df['text']     label = df['label'] 
 
    # Create a dataset from the DataFrame and apply the parse_data() function     payloads=df     payloads=payloads.fillna('').values     payloads=payloads[payloads.any(1)!=''] 
    payload = lb.transform(list(payloads[0,0])) 
 
    if len(payloads[0,0]) < max_len: 
        padding_vec = np.full((max_len-len(payloads[0,0]), num_classes), -1)         payload = np.concatenate((payload, padding_vec))     payload = payload[:max_len]     x = payload     y = payloads[0,1]     for i in range(1,len(df)): 
        print(i,end=" ")         payload = lb.transform(list(payloads[i,0]))         if len(payloads[i,0]) < max_len: 
            padding_vec = np.full((max_len-len(payloads[i,0]), num_classes), -1)             payload = np.concatenate((payload, padding_vec)) 
 
        payload = payload[:max_len]         x = np.append(x,payload,axis=0)         y = np.append(y,payloads[i,1]) 
     
    x = np.resize(x,(len(df),max_len,num_classes))      with open(filename[:-4]+'_x.npy', 'wb') as f:         np.save(f, x)      with open(filename[:-4]+'_y.npy', 'wb') as f: 
        np.save(f, y)
