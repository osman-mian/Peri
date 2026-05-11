#Author: Osman Ali Mian

import sys;
from utils import *
import gc #garbage collector
import numpy as np;

#project imports
from PeriGES import GESPerihelion;
from Environment import Environment
from GesX import GesX


def main():
	#sample path
	filenames 	=   [];
	envs		=	[]	#will store one environment per file name
	experiment 	= "reged15"
	
	for i in range(1,11):
		filenames.append("./data/"+experiment+"/experiment100/data"+str(i)+".txt")
		print(filenames[-1])

	alg = {'aic':0, 'bic':1, 'globe':2}
	score_ = alg['globe']	

	#load files
	for fn in filenames:								
		idata=Standardize(LoadData(fn.strip()));
		algorithm=GesX(idata.shape[1],idata,score_)
		envs.append(Environment(idata,algorithm));

	#first learn locally
	for env in envs:
		env.learn()


	#then optimize over worst-case regret
	sub_communication = 1.0 #use all environments in each iteration
	peri 		= GESPerihelion(envs,sub_sample=sub_communication);	
	network		= peri.Optimize();


	#calculate statistics
	truth_path			= "./data/"+experiment+"/experiment100/data1_truth.txt"
	ground_truth 		= LoadGroundTruth(truth_path,idata.shape[1],'\t');	
	SHD,F1				= Evaluate(getCpDag(ground_truth),getCpDag(network));		#SHD is calculated over skeleton


	#display
	print("Predicted CPDAG: \n", getCpDag(network))
	print("True CPDAG: \n", getCpDag(ground_truth))
	print("SHD:   ", SHD)
	print("F1:    ", F1)
	print("Total: ", np.sum(network))

	gc.collect()


main()









