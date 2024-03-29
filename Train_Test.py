#!/usr/bin/env python 
# coding: utf-8 
 
# In[1]: 
 
 
import tensorflow as tf from tensorflow.keras import metrics from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import AdditiveAttention ,Attention,Dense, Input,dot,Activation,Lambda,Concatenate,concatenate,Dropout,Embedding, 
Flatten, Conv1D, MaxPooling1D,AveragePooling1D 
,LSTM,Masking,Bidirectional,GlobalAveragePooling1D,GRU, Permute, multiply 
 
# from tensorflow.compat.v1.keras.layers import CuDNNGRU from sklearn.preprocessing import LabelBinarizer as lbe from tensorflow.keras import regularizers from tensorflow.keras.models import Model 
 
import numpy as np import pandas as pd 
 
import os from contextlib import redirect_stdout import matplotlib.pyplot as plt 
 
import datetime 
import pickle 
 
 
# In[2]: 
 
 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 
 
 
# # MODEL 
 
# In[4]: 
 
 
 
model = Sequential() 
 
 
model.add(Masking(mask_value=-1, input_shape=(4000,98))) 
 
model.add(Conv1D(filters=128, kernel_size=5, activation='relu')) 
 
model.add(AveragePooling1D(pool_size=5)) 
 
model.add(Conv1D(filters=128, kernel_size=3, activation='relu')) 
 
model.add(AveragePooling1D(pool_size=3)) 
# model.add(Conv1D(filters=128, kernel_size=2, activation='relu')) 
 
# model.add(AveragePooling1D(pool_size=3)) 
 
 
model.add((GRU(units=256, return_sequences=True))) 
 
model.add((GRU(units=128,   return_sequences=True))) 
 
model.add((GRU(units=64))) 
 
model.add(Dense(32, activation='relu')) 
 
model.add(Dense(1, activation='sigmoid')) 
 
 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) 
 
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',tf.metrics.Precision(),tf.metrics.AUC()]) 
 
 
# In[9]: 
 
 
 
model.summary() 
 
# In[10]: 
 
 
data_files = [] valFiles = [] 
 
noOfFiles=14 
 
for i in range(noOfFiles): 
    
data_files.append([['assets/data/data_{}_x.npy'.format(i)],['assets/data/data_{
}_y.npy'.format(i)]]) 
     
     
for i in range(noOfFiles,noOfFiles+2): 
        
valFiles.append([['assets/data/data_{}_x.npy'.format(i)],['assets/data/data_{}
_y.npy'.format(i)]]) 
 
     
data_files,valFiles 
 
 
# In[11]: 
 
 
def data_generator(file_list, batch_size):     while True:         for file in file_list: 
            x_train = np.load(file[0][0])             y_train = np.load(file[1][0])             num_samples = x_train.shape[0]             for i in range(0, num_samples, batch_size): 
                yield x_train[i:i+batch_size], y_train[i:i+batch_size] 
 
 
# In[12]: 
 
 
def lr_schedule(epoch): 
    lr = 0.001     if epoch > 10:         lr =  0.0001     return lr 
 
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule) 
 
 
# In[13]: 
 
 
from keras.callbacks import EarlyStopping 
 
batch_size = 128 
num_samples = 455 #no of emails in one file epch = 80 
 
early_stopping = EarlyStopping(monitor='val_loss', patience=20,verbose=1) 
 
history = model.fit(data_generator(data_files, batch_size), steps_per_epoch=len(data_files)*num_samples/batch_size,validation_data=d ata_generator(valFiles, batch_size), validation_steps=len(valFiles)*num_samples/batch_size, epochs=epch, callbacks=[early_stopping,lr_callback]) 
 
print("Epoch of early stop:", early_stopping.stopped_epoch) 
 
 
# In[14]: 
 
 
now = datetime.datetime.now() directory = now.strftime("%Y-%m-%d_%H-%M-%S") 
 
parent_dir = "assets/model/" 
   
path = os.path.join(parent_dir, directory) os.mkdir(path) 
 
 
# In[15]: 
 
 
 
with open(path + '/modelsummary.txt', 'w') as f:     with redirect_stdout(f):         model.summary() 
 
model.save(path + "/model.h5") 
 
 
with open(path + '/history.pkl', 'wb') as f: 
    pickle.dump(history.history, f) f.close() 
 
 
 
# Plot training & validation accuracy values plt.plot(history.history['accuracy']) plt.plot(history.history['val_accuracy']) plt.title('Model accuracy') plt.ylabel('Accuracy') plt.xlabel('Epoch') plt.legend(['Train', 'Validation'], loc='upper left') plt.savefig(path + '/acc.png') plt.show() 
# Plot training & validation loss values plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('Model loss') 
plt.ylabel('Loss') plt.xlabel('Epoch') plt.legend(['Train', 'Validation'], loc='upper left') plt.savefig(path + '/loss.png') 
 
plt.show() 
 
 
# In[16]: 
 
 
def testGen(file_list, batch_size):     while True:         for file in file_list: 
            x_train = np.load(file[0][0])             y_train = np.load(file[1][0])             num_samples = x_train.shape[0]             for i in range(0, num_samples, batch_size): 
                yield x_train[i:i+batch_size], y_train[i:i+batch_size] 
 
 
# In[17]: 
 
 
data_files = [] 
 
for i in range(16,20): data_files.append([['assets/data/data_{}_x.npy'.format(i)],['assets/data/data_{
}_y.npy'.format(i)]]) 
 
data_files 
 
 
# In[18]: 
 
 
batch_size = 32 num_samples = 455 # emails in each file result = model.evaluate(data_generator(data_files, batch_size), steps=len(data_files)*num_samples/batch_size) 
 
 
# In[19]: 
 
 
y_pred = [] y_true = [] 
 
 
j = 0 
 
for i in data_files:     if j==0:         j +=1 
        x = np.load(i[0][0])         y_pred = model.predict(x)         y_true = np.load(i[1][0])         del(x)     x = np.load(i[0][0]) 
     
    y_pred=np.concatenate((y_pred, model.predict(x)), axis=0)     y_true = np.concatenate((y_true, np.load(i[1][0])), axis=0)     del(x) 
     
     
     
 
 
# In[20]: 
 
 
from sklearn.metrics import precision_recall_curve from sklearn.metrics import roc_curve, auc 
 
precision, recall, thresholds = precision_recall_curve(y_true, y_pred) 
 
# Plot the precision-recall curve plt.plot(recall, precision) plt.xlabel('Recall') plt.ylabel('Precision') plt.title('Precision-Recall Curve') plt.savefig(path + '/precisionrecall.png') plt.show() 
 
 
 
# Compute the false positive rate, true positive rate, and AUC fpr, tpr, thresholds = roc_curve(y_true, y_pred) roc_auc = auc(fpr, tpr) 
 
# Plot the ROC curve plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' 
% roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') plt.xlabel('False Positive Rate') plt.ylabel('True Positive Rate') plt.title('Receiver operating characteristic (ROC) curve') plt.legend(loc="lower right") plt.savefig(path + '/roc.png') plt.show() 
 
with open(path + '/results.txt', 'w') as f: 
    f.write(str(result) + " epochs : " +str(early_stopping.stopped_epoch)) f.close() 
 
Extension model.py 
 
import pickle import numpy as np from keras.models import load_model def make_prediction(user_input, lb, model, max_len=4000, num_classes=98): 
    user_input = lb.transform(list(user_input))     if len(user_input) < max_len: 
        padding_vec = np.full((max_len - len(user_input), num_classes), -1)         user_input = np.concatenate((user_input, padding_vec))     user_input = user_input[:max_len]     predictions = model.predict(user_input.reshape((1, 4000, 98)))     return predictions 
 
 
if __name__ == "__main__":     with open("labels", "rb") as f:         lb = pickle.load(f)     model = load_model("model.h5")     print(make_prediction("Hello", lb, model)) 
 
Extension Server.py 
 
from flask import Flask, jsonify, request import pickle 
from model import make_prediction from keras.models import load_model from flask_cors import CORS 
 
app = Flask(__name__) 
CORS(app) 
 
# Load the trained model and vectorizer lb = pickle.load(open("labels", "rb")) model = load_model("model.h5") 
 
 
@app.route("/predict", methods=["GET"]) def predict(): 
    user_input = request.args.get("q")     prediction = make_prediction(user_input, lb, model)     score = prediction[0][0] 
 
    return jsonify({"prediction": float(score)}) 
 
@app.route("/predict", methods=["POST"]) def predictP(): 
    # Get the input data from the request body     request_data = request.get_json(force=True)     user_input = request_data['text'] 
     
    # Make the prediction and get the score     prediction = make_prediction(user_input, lb, model)     score = prediction[0][0] 
 
    # Return the prediction score as JSON     return jsonify({"prediction": float(score)}) if __name__ == "__main__":     app.run(debug=True, port=5000) 
 
