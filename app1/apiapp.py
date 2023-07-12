import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential
import pickle
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,BatchNormalization,Dropout,Input
from sklearn.tree import DecisionTreeClassifier

dt = pickle.load(open('dt.sav', 'rb'))
lg = pickle.load(open('lg.sav', 'rb'))
svm = pickle.load(open('svm.sav', 'rb'))
knn = pickle.load(open('knn.sav', 'rb'))

model=Sequential()
model.add(Input(shape=(7)))
model.add(Dense(73,activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dense(147,activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dense(37,activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dense(22,activation='softmax'))

model.load_weights('nn.h5')


from flask import Flask,request,jsonify,Blueprint, redirect, url_for



main = Blueprint('main', __name__)

@main.route('/')
def index():
    
    return "Hello"

@main.route('/predict1')
def predict():
	nc=request.form.get('nc')
	pc=request.form.get('pc')
	kc=request.form.get('kc')
	ph=request.form.get('ph')
	temp=request.form.get('temp')
	humidity=request.form.get('humidity')
	rain=request.form.get('rain')



	query=np.array([[float(nc),float(pc),float(kc),float(temp),float(humidity),float(ph),float(rain)]])
	DT_predict=dt.predict(query)

	knn_predict=knn.predict(query)

	SVM_predict=svm.predict(query)

	LG_predict=lg.predict(query)

	y_prob=model.predict(query)
	NN_predict=y_prob.argmax(axis=1)

	vote=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	maxVote=0
	labe=-1
	doc=0;
	y_ensemble=[]
	vote[knn_predict[doc]]+=1
	if maxVote<vote[knn_predict[doc]]:
		label=knn_predict[doc]
		maxVote=vote[knn_predict[doc]]
	vote[SVM_predict[doc]]+=1
	if maxVote<vote[SVM_predict[doc]]:
	    label=SVM_predict[doc]
	    maxVote=vote[SVM_predict[doc]]
	vote[NN_predict[doc]]+=1
	if maxVote<vote[NN_predict[doc]]:
	    label=NN_predict[doc]
	    maxVote=vote[NN_predict[doc]]
	vote[LG_predict[doc]]+=1
	if maxVote<vote[LG_predict[doc]]:
	    label=LG_predict[doc]
	    maxVote=vote[LG_predict[doc]]
	vote[DT_predict[doc]]+=1
	if maxVote<=vote[DT_predict[doc]]:
	    label=DT_predict[doc]
	    maxVote=vote[DT_predict[doc]]
	doc+=1;
	y_ensemble.append(label)

	res={'result':str(y_ensemble[0]) }
	return jsonify(res)