import gc 						#garbage collector
import numpy as np;
from alg.orion.orion_pp import Orion		#Orion++: parallelized, until convergence
#from alg.orion.orion_p import Orion		#Orion+: parallelized, single pass
#from alg.orion.orion import Orion		#Orion: serial, single pass
from alg.orion.utils import Standardize,LoadData

def main():
	try:
		fnames=["./data/intervention_det/experiment1/SI_4-5_6.txt","./data/intervention_det/experiment1/SI_0-2_1.txt","./data/intervention_det/experiment1/SI_0-1_0.txt"];
	
		D=[];
		#Load the Data 
		for file_name in fnames:
			print("Loading: ",file_name);
			idata=Standardize(LoadData(file_name));
			D.append(idata);

		variables=idata.shape[1];					#Number of variables

		#Provide the edge skeleton to search over,
		#If there is prior background knowledge over edges available, we can take it into account here
		E=[];
		for i in range(0,variables):
			for j in range(i+1,variables):
				E.append((i,j));

		print("Running Orion....");
		orion = Orion(D,E);							#Initialize Orion, the variant of Orion being used depends on the file you import on line #3, 4 and 5 of this file
		orion.optimize();							#Run Orion
		main_network,local_networks=orion.get_networks();	#Extract results


		print("Done...");
		print(main_network);						#Display result

		gc.collect();								#clear useless memory

	except Exception as e:
		print("Error in Orion: ");
		print(str(e));


main();
