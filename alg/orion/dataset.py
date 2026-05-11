import numpy as np;
from alg.orion.scorer import Scorer;
from copy import deepcopy;

class Dataset:


#---------- Functions for initialization--------------#
	def __init__(self,data):

		self.data=data;			#store the actual data
		self.V=data.shape[1];	#columns are the variables
		self.N=data.shape[0];	#rows are the records

		self.M= np.zeros(self.V,dtype=float);				#cost of storing mechanism for each variable
		self.DM= np.zeros(self.V,dtype=float);				#cost of storing residual for each variable

		self.adj_mat=np.zeros((self.V,self.V),dtype=int);	#local adj_matrix for this dataset

		self.fid_mat=np.zeros((self.V,self.V),dtype=int)-1;	#local data transformation matrix
		self.min_diff=np.zeros(self.V,dtype=float);			#need this for data given model score computation
		self.glb= Scorer(M=2);								#the scoring function

		#Call the init functions
		self.calc_min_diff();			
		self.init_data();

	def calc_min_diff(self):
		for v in range(self.V):
			variable=self.data[:,v];
			sorted_v =np.copy(variable)
			sorted_v.sort(axis=0);
			diff = np.abs(sorted_v[1]-sorted_v[0]);
			
			if diff==0: diff=np.array([9999]);
			
			for i in range(1,len(sorted_v)-1):
				curr_diff=np.abs(sorted_v[i+1]-sorted_v[i]);
				if curr_diff!=0 and curr_diff < diff:
					diff = curr_diff;
			self.min_diff[v]=diff;

	def init_data(self):
		for v in range(self.V):
			dt=[];			
			rows = self.N;
			dt.append(self.data[:,v].reshape(rows,-1)**0);
			source =np.hstack(dt);
			target=self.data[:,v];
			tot,m,dm = self.glb.ComputeScore(source,target,rows,self.min_diff[v],k=np.array([1]) );
			self.M[v]=m;
			self.DM[v]=dm;


#------------ Process Functions -------------#
	def test_edge(self,test_parent,child):
		tp=test_parent;
		m,dm,fid = self.calculate_cost(child,tp);

		delta_M= m - self.M[child];		#This should be +ve because more model parameters are added
		delta_DM= dm - self.DM[child];	#This should be -ve because more variance is explained
		acceptance_vote= 1

		#by checking that the sum of the two above is less than 0, we are asking if the overall cost has gone down, 
		#i.e the additional cost of adding parameters was covered by the cost of explained variance.
		#If this is not the case, we will not accept the edge and as a result the gain for this dataset would be zero
		if (delta_DM + delta_M) >= 0:
			acceptance_vote= 0
			delta_M=0;
			delta_DM=0;
			m=self.M[child];
			dm=self.DM[child];

		return delta_M, delta_DM,fid,acceptance_vote,m,dm;


	def test_del_edge(self,test_parent,child):
		if self.adj_mat[test_parent,child]==1:
			tp=test_parent;
			m,dm = self.calculate_neg_cost(child,tp);
			delta_M= m - self.M[child];		#This should be -ve because fewer model parameters are now considered
			delta_DM= dm - self.DM[child];	#This should be +ve because smaller variance is explained
		else:
			delta_M=0;
			delta_DM=0;
			m=self.M[child];
			dm=self.DM[child];

		acceptance_vote= 1
		return delta_M, delta_DM,acceptance_vote,m,dm;


	def add_edge_blind(self,parent,child,m,dm,fid,accept):
		if accept:
			if self.adj_mat[parent,child]==1:	#Just a sanity check to see if this edge is being re-added. This should never happen.
				print("Something is not right, edge ",parent," to ",child," is being re-added");
				print("Previous fid: ",self.fid_mat[parent,child]);
				print("New fid: ",fid);
				import ipdb;ipdb.set_trace();
			else:
				self.adj_mat[parent,child]=1;
				self.M[child]=m;
				self.DM[child]=dm;
				self.fid_mat[parent,child]=fid;


	def del_edge_blind(self,parent,child,m,dm,fid,accept):
		if self.adj_mat[parent,child]==1:
			self.adj_mat[parent,child]=0;
			self.M[child]=m;
			self.DM[child]=dm;
			self.fid_mat[parent,child]=0;

	def add_edgers(self,parent,child):
		m,dm,fid = self.calculate_cost(child,parent);

		delta_M= m - self.M[child];		#This should be +ve because more model parameters are added
		delta_DM= dm - self.DM[child];	#This should be -ve because more variance is explained
		
		if (delta_DM + delta_M) < 0:
			if self.adj_mat[parent,child]==1:	#Just a sanity check to see if this edge is being re-added. This should never happen.
				print("Something is not right, edge ",parent," to ",child," is being re-added");
				print("Previous fid: ",self.fid_mat[parent,child]);
				print("New fid: ",fid);
				import ipdb;ipdb.set_trace();
			else:
				self.adj_mat[parent,child]=1;
				self.M[child]=m;
				self.DM[child]=dm;
				self.fid_mat[parent,child]=fid;

	def calculate_cost(self,child,test_parent=None):
		parent_data=[];
		parent_transformations = []
		for pid in range(self.V):
			if pid!=child and self.adj_mat[pid,child]==1:
				parent_data.append(self.data[:,pid].reshape((-1,1)));
				parent_transformations.append(self.fid_mat[pid,child]);

		test_parent_data=self.data[:,test_parent].reshape((-1,1));
		child_data=self.data[:,child].reshape((-1,1));
		
		child_curr_bits=self.M[child]+self.DM[child];

		m,dm,fid=self.glb.GetEdgeAdditionCost(parent_data,test_parent_data,child_data,parent_transformations,self.min_diff[child],child_curr_bits); # GetEdgeAdditionCost(self,parents,candidate_parent,child,edge_parents_child):
		return m,dm,fid;

	def calculate_neg_cost(self,child,test_parent=None):
		parent_data=[];
		parent_transformations = []
		for pid in range(self.V):
			if pid!=child and pid!=test_parent and self.adj_mat[pid,child]==1:
				parent_data.append(self.data[:,pid].reshape((-1,1)));
				parent_transformations.append(self.fid_mat[pid,child]);

		child_data=self.data[:,child].reshape((-1,1));
		
		child_curr_bits=self.M[child]+self.DM[child];

		#GetCombinationCost(self, parents,fids,child,child_min_diff,child_curr_bits):
		a,a,a,m,dm=self.glb.GetCombinationCost(parent_data,parent_transformations,child_data,self.min_diff[child],child_curr_bits); 
		return m,dm;
		

#--------------Getter Functions---------------#

	def current_M(self):
		return self.M;

	def current_DM(self):
		return self.DM;

	def current_adj(self):
		return self.adj_mat;
