import numpy as np;
import os;
from alg.orion.dataCleaner import DataCleaner

def Standardize(variables):
	mu_ = np.mean(variables,axis=0);
	sdev_ = np.std(variables,axis=0);
	nvariables = (variables - mu_) / sdev_;
	n_vars=DataCleaner().CleanMat(nvariables,3); 
	return n_vars;

def LoadData(filename):
	with open(filename,'r') as file:
		k = file.readlines();

		dims = len(k[1].split(','));
		recs = len(k)-1;
		variables= np.zeros((1,dims));


		for i in range(1,recs):
			if 'nan' not in k[i].lower():
				line = k[i].split(',');
				temp=np.zeros((1,dims));

				for j in range(0,dims):
					temp[0,j]=line[j].strip();

				variables=np.vstack((variables,temp));
			else:
				recs=recs-1;
		variables=np.delete(variables,0,0);

	return variables;

def LoadPartialData(filename,N=1000):
	with open(filename,'r') as myfile:
		k = [next(myfile) for x in range(N)]

		dims = len(k[1].split(','));
		recs = len(k)-1;
		variables= np.zeros((1,dims));

		for i in range(1,recs):
			if 'nan' not in k[i].lower():
				line = k[i].split(',');
				temp=np.zeros((1,dims));

				for j in range(0,dims):
					temp[0,j]=line[j].strip();

				variables=np.vstack((variables,temp));
			else:
				recs=recs-1;
		variables=np.delete(variables,0,0);

	return variables;
