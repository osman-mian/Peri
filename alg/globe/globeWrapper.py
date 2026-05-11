from alg.globe.node import Node;
from alg.globe.edge import Edge;
from alg.globe.slope import Slope;
from alg.globe.globe import Globe;
from alg.globe.utils import *
from alg.globe.dag import DAG;
from alg.globe.statsCalculator import StatsCalculator
from alg.globe.skeletonHandler import SkeletonHandler;
from alg.globe.logger import Logger
import numpy as np;
import sys
import time as time
import os;
import gc;
from datetime import datetime

class GlobeWrapper:

	def __init__ (self,max_int=2,log_results=False,vrb=False):
		self.vars=np.zeros((5,5));
		self.M=max_int;
		self.log_path="./logs/log_"+ str(datetime.now(tz=None)).replace(' ','_')+ ".txt";
		self.log_flag= log_results;
		self.verbose=vrb;
		self.filename="";
		if self.log_flag:
			print("Saving results to: ",self.log_path)


	def setData(self,D):
		self.filename="None"
		self.vars=D;

	def loadData(self,filename):
		with open(filename,'r') as file:
			k = file.readlines();

			dims = len(k[1].split(','));
			recs = len(k)-1;
			dt = np.dtype('Float64')
			variables= np.zeros((1,dims),dtype=dt);


			for i in range(1,recs):
				if 'nan' not in k[i].lower():
					line = k[i].split(',');
					temp=np.zeros((1,dims),dtype=dt);

					for j in range(0,dims):
						temp[0,j]=line[j].strip();

					variables=np.vstack((variables,temp));
				else:
					recs=recs-1;
			variables=np.delete(variables,0,0);
		self.filename=filename;
		self.vars=variables;

	def run(self):
		normalized_vars=Standardize(self.vars);
		recs = normalized_vars.shape[0];
		dim = normalized_vars.shape[1];
		headers=[i for i in range(0,dim)];
		inclusive_model=True;
		point_model = True;

		slope_ = Slope();
		globe_ = Globe(dims=dim,M=self.M);

		logger = Logger(self.log_path,log_to_disk=self.log_flag,verbose=self.verbose);
		logger.Begin();
		logger.WriteLog("BEGIN LOGGING FOR FILE: "+self.filename);
		Edges = [[None for x in range(dim)] for y in range(dim)];
		Final_graph = [[None for x in range(dim)] for y in range(dim)];
		for k in range(dim):
			for j in range(dim):
				Edges[k][j]=Edge(-1,[],[],0);

		Nodes = [];
		for i in range(0,dim):	
			Nodes.append(Node(normalized_vars[:,i].reshape(recs,-1),globe_));

		undirected_edges=[];
		for i in range(0,dim):
			for j in range(i+1,dim):
				undirected_edges.append((i,j));
		

		#*******************************************************************#
		sh = SkeletonHandler(slope_,globe_,logger);
		pq,ri =sh.RankEdges(undirected_edges,Nodes,Edges,Final_graph);	
		gc.collect();

		graph_ = DAG(globe_,Nodes,Final_graph,Edges,headers,logger,pq,ri);
		graph_.ForwardSearch();
		graph_.BackwardSearch();
		gc.collect();
		logger.WriteLog("END LOGGING FOR FILE: "+self.filename);
		logger.End();

		network = np.eye(dim)*0
		for i in range(0,dim):
			for j in range(0,dim):
				if Final_graph[i][j] is not None:
					network[j,i]=1;	#need to flip indices (i,j to j,i) here because GLobe stores adjaceny matrix from Child to Parent
		
		tot_bits=[];
		for node in Nodes:
				tot_bits.append(node.GetCurrentBits());
		
		return network,tot_bits

