from peri_score import PeriScore;
import ges.main as ges
import numpy as np
import concurrent.futures
import gc;

class GESPerihelion:

	def __init__(self,env,sub_sample=1.0,max_processes=4,ln=0.0):
		self.V=env[0].V;
		self.max_proc=max_processes
		self.environments = env;
		self.subsamp=sub_sample
		self.lap_noise=ln
		
	def Optimize(self):
		D=self.environments[0].D
		old_network= np.random.randint(0, 2, (D.shape[1], D.shape[1]))
		iters=1
		max_iters=10
		while True:
			if iters>max_iters:
				print("Terminating after ",max_iters," iterations")
				return network,0,rounds


			score_class		=  PeriScore(D,self.environments,self.subsamp,self.lap_noise)
			network,_	 	=  ges.fit(score_class,phases=['forward', 'backward'])
		
			if np.sum(network-old_network)==0:
				break

			for env in self.environments:
				env.set_network(network)	

			iters+=1;
			old_network=network
			gc.collect()
		
		return network



	
