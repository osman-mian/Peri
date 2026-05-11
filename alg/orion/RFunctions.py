import numpy as np
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
from rpy2 import robjects;
from rpy2.robjects.packages import importr
MARS = importr('earth');

import re;
rpy2.robjects.numpy2ri.activate()

def REarth(X,Y,M=1):
	row,col=X.shape;
	rX=r.matrix(X,ncol=col,byrow=False);
	rY=r.matrix(Y,ncol=1,byrow=True);
	coeffs=[];

	try:
		rearth=MARS.earth(x=rX,y=rY,degree=M);
	except:
		print("Singular fit encountered, retrying with Max Interactions=1");
		rearth=MARS.earth(x=rX,y=rY,degree=1);

	COEFF_INDEX=11;
	CUTS_INDEX=6;
	SELECTED_INDEX=7;
	RSS_INDEX=0;

	arg_count=np.size(rearth[COEFF_INDEX]);
	cut_count=np.size(rearth[SELECTED_INDEX]);

	z=str(rearth[COEFF_INDEX]);
	lines=z.split('\n');
	interactions=[];
	for i in range(2,len(lines)):
		val=1+ lines[i].count('*');		
		interactions.append(val);

	tst=str(rearth[COEFF_INDEX]);
	listed=re.split('\n|h\(x[0-9]{1,1000}\-|h\(|\-x[0-9]{1,1000}\)|\)',tst)	
	for vs in listed:
		try:
			if vs is not None:
				coeffs.append(float(vs.strip()));
		except ValueError:
			pass;

	sse=rearth[RSS_INDEX][0]
	return sse,[coeffs],np.array([cut_count]),interactions;


