from sklearn import linear_model;
from scipy.special import comb
import numpy as np;
import RFunctions as rf


class MARSRegression:

	def name(self):
		print("MARS Regression...");

	def __init__(self,V,M=2):
		self.M = M
		self.V = V

	def Combinator(self,M,k):
		sum=comb(M+k-1, M);
		if sum==0:
			#print(M,",",k," abnormal");
			return 0;
		return np.log2(sum);


	def FitSpline(self,source,target):
		sse,coeff,hinge_count,interactions = rf.REarth(source,target,self.M)
		score = self.Model_Length(np.copy(coeff[0])) 
		return sse,score,coeff,hinge_count,interactions;

	
	def AggregateHinges(self, hinges,k):
		cost=0;
		flag=1;

		for M in hinges:
			cost = self.logN(M) + self.Combinator(M,k);
		return 	cost;


	def Regress(self,source,target):
		target=target.reshape((-1,1));
		rows=target.shape[0];
		bias= np.ones((rows,1));
		
		if source.shape[1]>0:
			source=np.hstack((bias,source));
		else:
			source=bias;

		k=source.shape[1]; #Parents = no. of columns in source
		base_cost=self.Model_Length(np.array([k])) + k*np.log2(self.V);
		sse,model,coeffs,hinges,interactions = self.FitSpline(source,target);
		base_cost = base_cost + self.Model_Length(hinges)+ self.AggregateHinges(interactions,k);

		L_m=model+base_cost;
		L_d_m = self.DataGivenModel_Length(sse,rows)

		return L_m, L_d_m;

	def DataGivenModel_Length(self, sse,n):
		var = sse/n
		sigma = np.sqrt(var)
		sigmasq = sigma**2;
		if sse == 0.0 or sigmasq == 0.0:
			return np.array([0.0]);
		else:
			err = (sse / (2 * sigmasq * np.log(2))) + ((n/2) * np.log2(2 * np.pi * sigmasq)) #- n * self.logg(resolution)
			return max(err,np.array([0]));

	def logN(self,z):
		z = np.ceil(z);
		
		if z < 1 :
			return 0;
		else :
			log_star = np.log2(z);
			sum = log_star;
			
			while log_star > 0:
				log_star = np.log2(log_star);
				sum = sum+log_star;
			
			return sum + np.log2(2.865064)


	def Model_Length(self,param):
		Nans = np.isnan(param);
		
		if any(Nans):
			print ('Warning: Found Nans in regression coefficients. Setting them to zero...')
		param[Nans]=0;
		sum =0;
		for c in param:
			if np.abs(c)>1e-12:
				c_abs =  np.abs(c);
				c_dummy = c_abs;
				precision = 1;
				
				while c_dummy<1000:
					c_dummy *=10;
					precision+=1;
				sum = sum + np.log2(c_dummy) + self.logN(precision) + 1
		return sum;

