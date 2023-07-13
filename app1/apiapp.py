import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

dt = pickle.load(open('dt.sav', 'rb'))
lg = pickle.load(open('lg.sav', 'rb'))
svm = pickle.load(open('svm.sav', 'rb'))
knn = pickle.load(open('knn.sav', 'rb'))


from flask import Flask,request,jsonify,Blueprint, redirect, url_for



main = Blueprint('main', __name__)

@main.route('/')
def index():
    
    return "Hello"

@main.route('/predict1',methods=['POST'])
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
	s=""
	if str(y_ensemble[0])=='0':
		s='Apple'
	if str(y_ensemble[0])=='1':
		s='Banana'
	if str(y_ensemble[0])=='2':
		s='Backgram'
	if str(y_ensemble[0])=='3':
		s='Chickpea'
	if str(y_ensemble[0])=='4':
		s='Cocunut'
	if str(y_ensemble[0])=='5':
		s='Coffee'
	if str(y_ensemble[0])=='6':
		s='Cotton'
	if str(y_ensemble[0])=='7':
		s='Grapes'
	if str(y_ensemble[0])=='8':
		s='Jute'
	if str(y_ensemble[0])=='9':
		s='Kidenybeans'
	if str(y_ensemble[0])=='10':
		s='Lentil'
	
	if str(y_ensemble[0])=='11':
		s='Maize'
	if str(y_ensemble[0])=='12':
		s='Mango'
	if str(y_ensemble[0])=='13':
		s='Mothbeans'
	if str(y_ensemble[0])=='14':
		s='Mungbeans'
	if str(y_ensemble[0])=='15':
		s='Muskmelon'
	if str(y_ensemble[0])=='16':
		s='Orange'
	if str(y_ensemble[0])=='17':
		s='Papaya'
	if str(y_ensemble[0])=='18':
		s='Pegionpea'
	if str(y_ensemble[0])=='19':
		s='Pomegranate'
	if str(y_ensemble[0])=='20':
		s='Rice'
	if str(y_ensemble[0])=='21':
		s='Watermelon'
	
	res={'result':s }
	return jsonify(res)
