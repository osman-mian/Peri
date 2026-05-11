from alg.orion.dataset import Dataset
from alg.orion.TwoWayPriorityQueue import TwoWayPriorityQueue;
from scipy.special import comb
from alg.orion.graphUtil import GraphUtil;
from copy import deepcopy
import gc #garbage collector
import time;
import numpy as np;

class Orion:
	def __init__(self,data,edges=None,gt=None):
		self.data=data;
		self.pq = TwoWayPriorityQueue({});	#for storing candidates
		self.D=len(data);
		self.V=data[0].shape[1];
		self.E=[];		
		self.truth=gt;
		self.cache_book={};
		print("Running Orion, (serial, single pass)");
		if edges is not None:
			self.E=edges;
		else:
			for i in range(V):
				for j in range(i+1,V):
					self.E.append((i,j));

		#init dataset objects
		self.datasets=[];
		for i in range(self.D):
			self.datasets.append(Dataset(self.data[i]));

		self.network=np.zeros((self.V,self.V),dtype=int);	#global adj_matrix for the entire network

		self.init_edges();
		self.gu = GraphUtil();

	def get_networks(self):
		local_nets=[];
		for i in range(self.D):
			local_nets.append(self.datasets[i].current_adj());
		return self.network, local_nets

	def init_edges(self):
		for e in self.E:
			x1=e[0];
			x2=e[1];

			d1_2=self.calculate_delta(x1,x2);
			d2_1=self.calculate_delta(x2,x1);
			phi= d1_2 - d2_1;
			self.pq[(x1,x2)]=phi;
			self.pq[(x2,x1)]=-phi;



	def calculate_delta(self,parent,child):
		#Note to self: always use the new minus old standard.
		dms=[];
		ms=[];
		votes=[];
		for d in range(self.D):
			dataset=self.datasets[d];
			delta_m,delta_dm,fid,accept,m,dm = dataset.test_edge(parent,child);
			self.cache_book[(d,parent,child)]=(m,dm,fid,accept);
			dms.append(delta_dm);
			ms.append(delta_m);
			votes.append(accept);

		total_votes=np.sum(np.array(votes));
		#print(parent, " -> ",child);
		#print("votes: ",votes);
		#print("gains: ", np.array(dms) + np.array(ms));
		#print("****");
		if total_votes==0:
			#print("No one wanted: ",parent," -> ",child);
			return 0;

		#S1
		delta1=0;	#just the difference between two const scores;

		#S2
		parents = sum(self.network[:,child]);
		delta2= (self.logN(parents+1)-self.logN(parents)) + (self.logg(comb(self.V,parents+1)) - self.logg(comb(self.V,parents)) ) ;
		
		#S3
		delta3=0;
		for d in range(self.D):
			adj = np.sum(self.network - self.datasets[d].current_adj(),axis=0);			#global network is always supposed to have more edges than local, so this op will return positive values;
			v_int = np.where(adj>0)[0];													#this should give you the variables that are missing incoming edges, i.e intervened in this dataset 	
			v_ = adj[child];															#This is the no. of intervened parents in the current dataset for the child variable.
			if votes[d]:		#vote was in favor
				d3_1= 0;																#no additional intervention parents added so no change of cost
				d3_2= self.logg(parents+1)-self.logg(parents)							#one additional parent acquired
				d3_3= self.logg(comb(parents+1,v_)) - self.logg(comb(parents,v_));		#one additional parent added but no new intervention parents to encode

			elif not votes[d]:	#vote was against
				if child not in v_int:													#Meaning this child was not marked intervened previously
					#d3_1= self.logg(self.V);
					d3_1= self.logg(comb(self.V,len(v_int)+1)) - self.logg(comb(self.V,len(v_int)));	#among V variables now v_int+1 are intervened and need to be identified.
				else:
					d3_1=0;																#Child was already intervened, nothing changes
				d3_2= self.logg(parents+1)-self.logg(parents)							#one additional parent acquired
				d3_3= self.logg(comb(parents+1,v_+1)) - self.logg(comb(parents,v_));	#one additional parent added and one new intervention parents to encode

			delta3=delta3+ (d3_1 + d3_2 + d3_3);

		#S4 and S5
		delta4=np.sum(np.array(ms,dtype=float));
		delta5=np.sum(np.array(dms,dtype=float));
		
		global_delta = delta1 + delta2 + delta3;
		local_delta  = delta4 + delta5;

		delta_total= global_delta + local_delta;
		#print("Total: ",delta_total);
		return min(0,delta_total);

	def print_st(self):
		#return;
		qq=deepcopy(self.pq);
		v=-1;
		print(self.truth);
		print("------------------")
		while qq.hasMore() and v<0:
			k = qq.next_best();
			v = qq[k];
			print(k,": ",v);
			qq.removeEntry(k);
		print("------------------")

	def optimize(self):
		if not self.validate():
			print("Unable to run Orion, there is a problem with input validation");
			return;
		adds=0;
		pq=self.pq;
		#import ipdb;ipdb.set_trace();
		while pq.hasMore():
			k = pq.next_best();
			v = pq[k];

			parent=k[0];	
			child =k[1];	
			
			pq.removeEntry(k)

			#test cycle
			flag1= self.gu.CausesCycle(self.network,child,parent);
			if flag1:
				#print("Cyclic(",v,"): ",parent," -> ", child);
				continue;
			
			#test significance
			flag2= v > -10
			if flag2:
				#print("Insignificant(",v,"): ",parent," -> ", child);
				continue;

			self.network[parent,child]=1;
			for d in range(self.D):
				m,dm,fid,accept=self.cache_book[(d,parent,child)]
				self.datasets[d].add_edge_blind(parent,child,m,dm,fid,accept);
			adds=adds+1;
			other_edges= 2 - self.network[:,child] - self.network[child,:]	#list of all parents/children, this will be 2 if there is no incoming or outgoing edge to a node from the current node
			for p in range(len(other_edges)):
				if p==child or other_edges[p]!=2:		#meaning either parent or child edge already exists
					continue
				
				d1_2=self.calculate_delta(p,child);
				d2_1=self.calculate_delta(child,p);
				phi= d1_2 - d2_1;
				self.pq[(p,child)]=phi;
				self.pq[(child,p)]=-phi;

			gc.collect()

		print("Total Added: ",adds);
#--------------------- Utility Functions------------------------#

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
		
	def logg(self,x):
		if x == 0:
			return 0
		else:
			return np.log2(x)

	#periodically add all the validations here
	def validate(self):
		self.colFlag=True;
		for i in range(self.D):
			if self.data[i].shape[1] != self.data[0].shape[1]:
				self.colFlag=False;
				return False;

		return True;

#------------------Actual Score--------------#
	def calculate_bits(self):
		#S1
		bits=self.logN(self.D) + self.logN(self.V) + self.D * self.logg(self.V)

		#S2
		for v in range(self.V):
			p = np.sum(self.network[:,v]);
			bits = bits + self.logN(p) + self.logg(comb(self.V,p));

		#S3
		for d in range(self.D):
			#calculate interventions
			adj = self.network - self.datasets[d].current_adj();	#global network is always supposed to have more edges than local, so this op will return positive values;
			adj = np.sum(adj,axis=0);								#take column sum because we are after incoming edges that are missing locally
			v = np.where(adj>0)[0];									#this should give you the variables that are missing incoming edges
			bits = bits + self.logg(comb(self.V,len(v)));
			
			#iterate over intervened v's
			for v_ in v:
				p = np.sum(self.network[:,v_]);
				if p == adj[v_]:	#hard intervention
					bits=bits+1;
				else:
					bits= bits  + 1  + self.logg(adj[v_]) + self.logg(comb(p,v_));
			
		#S4 and S5
		for dataset in self.datasets:
			bits=bits + sum(dataset.current_M()) + sum(dataset.current_DM())

		return bits;
