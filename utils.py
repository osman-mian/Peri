import os;
import numpy as np;

from dataCleaner import DataCleaner
import ges.utils as gutils

def getCpDag(zh):
	adj_list=gutils.pdag_to_cpdag(zh)
	return adj_list

def Evaluate(zg,zh):
	shd = 	int(np.sum(np.abs(zg-zh)))
	F1	=	findF1score(zg,zh)
	return shd,np.round(F1,4);


def findF1score(zg,zh):
	g = np.array(zg,dtype=bool)
	h = np.array(zh,dtype=bool)

	tp = np.sum(g * h)
	fp = np.sum(g * np.logical_not(h))
	fn = np.sum(np.logical_not(g) * h)

	if tp==0:
		return 0

	prec = tp / (tp+fp)
	rec = tp / (tp+fn)

	f1= (2*prec*rec) / (prec+rec);
	return f1;

def LoadGroundTruth(fname,node_size,sep='\t'):
	alpha=None;
	if not os.path.exists(fname):
		print ('Ground Truth file not found, nothing to compare for ',fname);
	else:
		alpha=np.eye(node_size) * 0;				
		with open(fname,'r') as file:
			k = file.readlines();
			for i in range(len(k)):
				line_ = k[i].split(sep);
				#print("THIS IS K:.........>",k[i]);
				s=int(line_[0].strip());
				t=int(line_[1].strip());
				alpha[s,t]=1;

				
	return np.array(alpha,dtype=np.int8);
 

def Standardize(variables):
		mu_ = np.mean(variables,axis=0);
		sdev_ = np.std(variables,axis=0);
		nvariables = (variables - mu_) / sdev_;
		n_vars=DataCleaner().CleanMat(nvariables,3); return n_vars;
		#return nvariables;


def GetRandom(vars_,ratio=1,count=0):
	#return vars;
	size = vars_.shape[0];
	if count==0:
		req = size* ratio;
	else:
		req = count;
	filtered = vars_[np.random.choice(size, req, replace=False), :];
	
	return filtered;




def LoadPartialData(filename,N=1000):
	with open(filename,'r') as myfile:
		k = [next(myfile) for x in range(N)]

		dims = len(k[1].split(','));
		recs = len(k)-1;
		#dt = np.dtype('Float64')
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

def LoadData(filename):
	with open(filename,'r') as file:
		k = file.readlines();

		dims = len(k[1].split(','));
		recs = len(k)-1;
		#dt = np.dtype('Float64')
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

def LoadHeader(fname,offset=0):
	alpha={};
	rev_alpha={};
	i=0;
	#print ('Probing: ',fname);
	
	if not os.path.exists(fname):
		for k in range(101):
			alpha[k]=str(k+offset);
			rev_alpha[alpha[k]]=k;
		print ('Header file not found, using numbering...');
		return alpha,rev_alpha;
				
	with open(fname,'r') as file:
		k = file.readlines();
		for i in range(len(k)):
			alpha[i]=k[i].strip();
			rev_alpha[alpha[i]]=i;
	#import ipdb;ipdb.set_trace();
	return alpha,rev_alpha;

def LoadGroundT(fname,id=11):
	alpha=[];
	i=0;
	#print ('Probing: ',fname);
	
	if not os.path.exists(fname):
		print ('Ground Truth file not found, nothing to compare');
		return alpha,False;
				
	with open(fname,'r') as file:
		k = file.readlines();
		for i in range(len(k)):
			if id==11:
				line_ = k[i].split(' ');
			else:
				line_ = k[i].split('\t');
			#print(k[i]);
			s=line_[0].strip();
			t=line_[1].strip();
			alpha.append((s,t));
			
	
	#import ipdb;ipdb.set_trace();
	return alpha,True;



def Plot2d(source,target,fname):
	plt.ioff();
	plt.figure()
	plt.clf();
	
	plt.plot(source,target,'b.');
		
	plt.savefig(fname);
	
	plt.clf();
	plt.close();
