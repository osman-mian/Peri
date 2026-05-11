import numpy as np;
import gc
from copy import deepcopy
from queue import PriorityQueue
import ges.utils as utils
import concurrent.futures
class Environment:
	def __init__(self,data,alg):
		self.D=data;
		self.V=self.D.shape[1];
		self.network=set();
		self.nodewise_score=[];
		self.total_regret= 0
		self.alg=alg;
		self.empty_score=[];
		self.reg_cache={};
		for i in range(self.V):
			self.reg_cache[i]={};
		self.max_proc=2
		self.learned=False

	def learn(self,debug=False):
		netw,nws = self.alg.learn(self.D)	#this will return a pdag
		self.network=netw
		ids_ = [(np.flatnonzero(self.network[:,v]) , v ) for v in range(self.V)]
		self.nodewise_score=deepcopy(nws)
		self.learned=True;
		return self.network;

	def score(self,parents,child):
		if not self.learned:
			self.learn();

		parents.sort()
		#import ipdb;ipdb.set_trace()
		if tuple(parents) not in self.reg_cache[child]:
			curr_score=self.alg.calculateScoreModular(parents,child,self.D);
			res = curr_score - self.nodewise_score[child]
			self.reg_cache[child][tuple(parents)]=res;
		else:
			res=self.reg_cache[child][tuple(parents)]
		return res


	#'''
	def set_network(self,netw):
		self.network=utils.pdag_to_cpdag(netw);
		ids_ = [(np.flatnonzero(self.network[:,v]) , v ) for v in range(self.V)]
		self.nodewise_score=np.array([None for v in range(self.V)])
		v=0
		with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_proc) as executor:
			scs=executor.map(self.concurrent_score,ids_)
			for p in scs:
				self.nodewise_score[v]=p[0]
				self.reg_cache[v][p[1]]=p[0]
				v=v+1;
		self.learned=True;

	#'''



	def concurrent_score(self,pc):
		parents=pc[0];
		parents.sort()
		child=pc[1]
		if tuple(parents) not in self.reg_cache[child]:
			res=self.alg.calculateScoreModular(parents,child,self.D);
		else:
			res=self.reg_cache[child][tuple(parents)]
		return  res,tuple(parents)

	#return sum of all node-wise regrets for a specific input network
	# regret should be in (-inf, 0]. Any score that is plugged in should be a score to be maximized.
	# if the input score needs to be minimized, multiply it by -ve sign in your instantiation of 
	# decomposable score before returning the final answer
	'''
	def global_regret(self,adj_list,debug=False):
		adj_list=utils.pdag_to_cpdag(adj_list)
		ids_=[(np.flatnonzero(adj_list[:,child]),child) for child in range(self.V)];
		adj_score=np.array([None for v in range(self.V)])
		v=0;
		with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_proc) as executor:
			adjs = executor.map(self.concurrent_score,ids_)
			for p in adjs:
				adj_score[v]=p[0]
				self.reg_cache[v][p[1]]=p[0]
				v=v+1;
		regret = adj_score - self.nodewise_score
		gc.collect()
		return np.sum(regret);

	#'''



