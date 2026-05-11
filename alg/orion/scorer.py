import numpy as np;
import alg.orion.RFunctions as rf
from copy import deepcopy;
from alg.orion.combinator import Combinator;
from alg.orion.dataTransformer import DataTransformer

class Scorer:
	def __init__(self,M=2):
		self.Transformer= DataTransformer(True);
		self.terms = {0:1,1:2,2:3,3:1,4:1,5:1,6:4,7:1,8:1}
		self.F=9;
		self.M=M;
	
	def GetEdgeAdditionCost(self,parents,candidate_parent,child,fids,child_min_diff,child_curr_bits):
		new_parents = deepcopy(parents);
		new_parents.append(candidate_parent);
		best_compression= 10;
		best_fids = -1;

		new_edges  = fids;
		for f_id in range(self.F):
			new_edges.append(f_id);
			perc_reduction,abs_reduction,total_cost,m,dm = self.GetCombinationCost(new_parents,new_edges,child,child_min_diff,child_curr_bits)
			new_edges.remove(f_id);		
			
			if perc_reduction < best_compression and f_id!=-1:
				best_compression= perc_reduction;
				best_fids = f_id
				best_m=m;
				best_dm=dm;
		return best_m,best_dm,best_fids;
	
	def GetCombinationCost(self, parents,fids,child,child_min_diff,child_curr_bits):
		dt=[];
		parent_count = np.array([len(parents)]);
		
		rows=child.shape[0]
		dt.append(child.reshape(rows,-1)**0);
		for i in range(len(parents)):
			dt.append(self.Transformer.TransformData(parents[i],fids[i]));

		source =np.hstack(dt);
		target=child;

		total_cost,m,dm = self.ComputeScore(source,target,rows,child_min_diff,parent_count);
		perc_reduction = total_cost/child_curr_bits;
		abs_reduction = child_curr_bits - total_cost#max(0,child_curr_bits - total_cost);
		return perc_reduction,abs_reduction,total_cost,m,dm;
	
	def ComputeScore(self,source,target,rows,mindiff,k,show_graph=False):
		sse,model_coeff,coeffs,hinges,interactions = self.FitSpline(source,target,self.M);			#Fit a spline
		m = model_coeff + self.model_score(hinges)+ self.AggregateHinges(interactions,k);		#derive the model cost
		dm=self.gaussian_score_emp_sse(sse,rows,mindiff);										#derive the data given_model cost
		return (m+dm),m,dm;
		
	def AggregateHinges(self, hinges,k):
		cost=0;
		for M in hinges:
			cost = cost + self.logN(M) + Combinator(M,k) + M*np.log2(self.F);
		return 	cost;

	#Functions for fitting spline and scoring
	def FitSpline(self,source,target,M=2,temp=False):
		sse,coeff,hinge_count,interactions = rf.REarth(source,target,M)
		score = self.model_score(np.copy(coeff[0])) 
		return sse,score,coeff,hinge_count,interactions;

	def model_score(self,coeff):
		Nans = np.isnan(coeff);
		if any(Nans):
			print ('Warning: Found Nans in regression coefficients. Setting them to zero...')
		coeff[Nans]=0;
		sum =0;
		for c in coeff:
			if np.abs(c)>1e-12:
				c_abs =  np.abs(c);
				c_dummy = c_abs;
				precision = 1;
				
				while c_dummy<1000:
					c_dummy *=10;
					precision+=1;
				sum = sum + self.logN(c_dummy) + self.logN(precision) + 1
		return sum;
	
	def gaussian_score_emp_sse (self,sse, n,resolution):
		sigmasq = sse / n;
		if sse == 0.0 or sigmasq == 0.0:
			return np.array([0.0]);
		else:
			err = (sse / (2 * sigmasq * np.log(2))) + ((n/2) * self.logg(2 * np.pi * sigmasq)) - n * self.logg(resolution)
			return max(err,np.array([0]));
		
	def logg(self,x):
		if x == 0:
			return 0
		else:
			return np.log2(x) 

	def logN(self,z):
		z = np.ceil(z);
		
		if z < 1 :
			return 0;
		else :
			log_star = self.logg(z);
			sum = log_star;
			
			while log_star > 0:
				log_star = self.logg(log_star);
				sum = sum+log_star;
			
			return sum + self.logg(2.865064)

