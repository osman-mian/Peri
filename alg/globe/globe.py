import numpy as np;

from alg.globe.slope import Slope;
from alg.globe.combinator import Combinator;
from alg.globe.edge import Edge;
from alg.globe.sampler import Sampler;
from alg.globe.dataTransformer import DataTransformer
from alg.globe.dataCleaner import DataCleaner;
class Globe:

	def __init__(self,dims=0,M=3):
		self.slope_ = Slope();
		self.sampler = Sampler();
		self.Transformer= DataTransformer(True);
		self.terms = {0:1,1:2,2:3,3:1,4:1,5:1,6:4,7:1,8:1}
		self.F=1;
		self.V=dims;
		self.M=3;
		#print("Max interactions set to degree ",self.M);
	
	def GetEdgeAdditionCost(self,parents,candidate_parent,child,edge_parents_child):
		new_parents = parents;
		new_parents.append(candidate_parent);
		
		best_compression= 9999;
		best_bits = 99999;
		best_fids = -1;
		best_absolute = np.array([99999]);
		coeffs=[];
		for f_id in range(self.F):
			
			new_edges  = edge_parents_child;
			ed=Edge(f_id,[],[],0);
			new_edges.append(ed);
			gain_in_bits,new_bits,coeff,absolute_bits = self.GetCombinationCost(new_parents,new_edges,child)
			coeff_terms = self.terms[f_id];
			new_edges.remove(ed);
			
			if gain_in_bits < best_compression and f_id!=-1:
				best_compression= gain_in_bits;
				best_bits = new_bits;
				best_fids = f_id
				coeffs = coeff[0][len(coeff[0])-coeff_terms:len(coeff[0])];
				best_absolute=absolute_bits;
		
		new_parents.remove(candidate_parent);
		return best_compression,best_bits,best_absolute,best_fids,coeffs;
	
	def GetAverageCompression(self ,parents1,parent2,child,edge_parents1_child,edge_parent2_child,max_iter=100):
		rows=child.GetData().shape[0]
		gains = np.zeros((max_iter,self.F));
		parent_count = np.array([len(parents1)]) + 1;
		dt=[];
		dt.append(child.GetData().reshape(rows,1)**0);
		for i in range(len(parents1)):
			dt.append(self.Transformer.TransformData(parents1[i].GetData(),edge_parents1_child[i].GetFunctionId()));
		
		running_average=0;
		thresh=10;
		tolerance=0;	
		for i in range(max_iter):
			mutated_data = self.sampler.Mutate(parent2.GetData());
			for fid in range(self.F):
				dt2 = dt;
				app_var=self.Transformer.TransformData(mutated_data,fid);
				dt2.append(app_var);

				source =np.hstack(dt2);
				target=child.GetData();
				
				new_bits,coeffs = self.ComputeScore(source,target,rows,child.GetMinDiff(),parent_count);
				gains[i,fid] = max(0,child.GetCurrentBits() - new_bits);
				del dt2[-1];
				
			if np.abs(running_average-np.mean(gains))<thresh:
					tolerance=tolerance+1;
					if tolerance > 200:
						break;
			else:
					tolerance=0;
			running_average=np.mean(gains);
		return np.mean(gains);
		
	def GetCombinationCost(self, parents,edge_parents_child,child,debug=False):

		dt=[];
		parent_count = np.array([len(parents)]);
		
		rows=child.GetData().shape[0]
		dt.append(child.GetData().reshape(rows,-1)**0);
		for i in range(len(parents)):
			dt.append(self.Transformer.TransformData(parents[i].GetData(),edge_parents_child[i].GetFunctionId()));

		source =np.hstack(dt);
		target=child.GetData();

		new_bits,coeff = self.ComputeScore(source,target,rows,child.GetMinDiff(),parent_count);
		gain_in_bits = new_bits/child.GetCurrentBits();
		absolute_gain_in_bits = max(0,child.GetCurrentBits() - new_bits[0]);


		return gain_in_bits,new_bits,coeff,np.array([absolute_gain_in_bits]);
	
	def ComputeScore(self,source,target,rows,mindiff,k,show_graph=False):
		#print("Base Score")
		base_cost=self.slope_.model_score(k) + k*np.log2(self.V);
		#print("Fitting spline")
		sse,model,coeffs,hinges,interactions = self.slope_.FitSpline(source,target,self.M,show_graph);
		#sse,model,coeffs = self.slope_.FitModel(source,target,show_graph);
		#print("Agregating hinges")
		base_cost = base_cost + self.slope_.model_score(hinges)+ self.AggregateHinges(interactions,k);
		#print("Data given model")
		cost = self.slope_.gaussian_score_emp_sse(sse,rows,mindiff)+model+base_cost;
		#print("Done")
		return cost,coeffs;
		
	
	def AggregateHinges(self, hinges,k):
		cost=0;
		for M in hinges:
			cost += self.slope_.logN(M) + Combinator(M,k) + M*np.log2(self.F);
		return 	cost;
	
