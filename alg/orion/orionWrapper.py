import gc 						#garbage collector
import numpy as np;
from alg.orion.orion_pp import Orion		#Orion++: parallelized, until convergence
#from alg.orion.orion_p import Orion		#Orion+: parallelized, single pass
#from alg.orion.orion import Orion		#Orion: serial, single pass
from alg.orion.utils import Standardize,LoadData

class OrionWrapper:


	def __init__ (self,dt,e=None):
		self.D=dt
		#Provide the edge skeleton to search over,
		#If there is prior background knowledge over edges available, we can take it into account here
		self.E=e;
		if E is None: 
			for i in range(0,variables):
				for j in range(i+1,variables):
					E.append((i,j));

	def run(self):
		try:
			orion = Orion(self.D,self.E);							#Initialize Orion, the variant of Orion being used depends on the file you import on line #3, 4 and 5 of this file
			orion.optimize();								#Run Orion
			main_network,local_networks=orion.get_networks();	#Extract results
			gc.collect();								#clear useless memory
			return main_network

		except Exception as e:
			print("Error in Orion: ");
			print(str(e));

		return None



