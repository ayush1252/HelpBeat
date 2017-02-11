import sklearn
import pickle
import numpy as np
from random import randint
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
input_file = open("data_clean_imputed.csv","r")
lines = input_file.readlines()


CLASSIFICATION_TYPE = 2
NUM_PCA = 270;

X = []

for line in lines:
		tokens = line.strip().split(",")
		X.append(map(float, tokens[0:NUM_PCA]))

patientname="John Doe"
num=randint(0,NUM_PCA-1)
age=X[num][0]
sex=""
if(X[num][1]==0):
	sex="Male"
else:
	sex="Female"
height=X[num][2] #in cm
weight=X[num][3] #in kg

ypred=loaded_model.predict(X[num])
print ypred




