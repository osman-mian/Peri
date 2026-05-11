import numpy as np;
import sys;
#import .ges;
import ges.main as gs
from globe_score import MDLScore;
from RegscorePy import *
class GesX:
	def __init__(self,V,D,scorer=0):
		self.V=V;
		self.score_id=scorer
		self.D=D
		self.score_class=self.getScoreClass()
		self.ph=['forward', 'backward']

	def learn(self,D):
		network, _ = gs.fit(self.score_class,phases=self.ph)
		score=np.zeros(self.V);
		for v in range(self.V):
			child=v;
			pa=set();
			for p in range(self.V):
				if network[p,v]!=0:
					pa.add(p);
			score[v]=self.score_class.local_score(child,pa);
		return network,score;

	def scoreEmpty(self,D):
		score=np.zeros(self.V);
		for v in range(self.V):
			child=v;
			pa=set();
			score[v]=self.score_class.local_score(child,pa);
		return np.sum(score);

	def calculateScoreModular(self,parents,child,D=None,debug=False):
		print(parents) if debug else None;
		pa=set(parents)
		score=self.score_class.local_score(child,pa);
		return score


	def getScoreClass(self):
		#print("Initializing GES")
		if self.score_id==2:
			#print("Using MDL")
			return MDLScore(self.D);
		elif self.score_id==0:
			#print("Using AIC")
			return gs.GaussObsL0Pen(self.D,lmbda=1,method='raw')	#parameter penalization will not scale with number of points
		else:		#if any other value given, default to BIC
			#print("Using BIC")
			return gs.GaussObsL0Pen(self.D,lmbda=0.5 * np.log(self.D.shape[0]),method='raw') #parameter penalization scales with log(n)


		
