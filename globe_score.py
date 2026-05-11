# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
"""

import numpy as np
from ges.decomposable_score import DecomposableScore
#from alg.globe.globeWrapper import GlobeWrapper;
from alg.globe.globe import Globe
# --------------------------------------------------------------------
# l0-penalized Gaussian log-likelihood score for a sample from a single
# (observational) environment


class MDLScore(DecomposableScore):
	"""
	Implements MDL score from Mian et al. (2021), Discovering Fully Oriented
	Causal Networks, AAAI-2021.

	"""

	def __init__(self, data, cache=True, debug=0):
		"""Creates a new instance of the class.
		Parameters
		----------
		data : numpy.ndarray
			the nxp matrix containing the observations of each
			variable (each column corresponds to a variable).
		cache : bool, optional
		   if computations of the local score should be cached for
		   future calls. Defaults to True.
		debug : int, optional
			if larger than 0, debug are traces printed. Higher values
			correspond to increased verbosity.
		"""
		if type(data) != np.ndarray:
			raise TypeError("data should be numpy.ndarray, not %s." % type(data))

		super().__init__(data, cache=cache, debug=debug)
		self.n, self.p 	= data.shape
		self.scorer 	= Globe(dims=self.p,M=2);
		self.pcache		= {};
		self.mindiff	= [1e-9 for i in range(self.p)]
		for i in range(self.p):
			self.mindiff[i]	= self.findMindiff(data[:,i]);

	def _compute_local_score(self, x, pa):
		score=0;
		pa=list(pa)
		pa.sort()
		if (tuple(pa),x) not in self.pcache:
			target 		= self._data[:,x]
			if len(pa)==0:
				source	= self._data[:,x].reshape(self.n,-1)**0;
				score	= self.calculateScore(source,target,self.mindiff[x],False);
			else:
				source 	= self._data[:,pa]
				score	= self.calculateScore(source,target,self.mindiff[x]);
		return -1*score


	def full_score(self,A0):
		score=0;
		for x in range(self.p):
			pa		=	np.nonzero(A0[:,x])[0].tolist()			#find parent ids'
			pa.sort()
			score	+=	self._compute_local_score(x, pa)		#calculate local score, and sum up

		return score

	def calculateScore(self,source,target,mindiff,append_intercept=True):
		intercept_col	= target.reshape(self.n,-1)**0
		source			= source.reshape(self.n,-1)

		if append_intercept:
			source		= np.hstack((intercept_col,source));
		
		cost,_	= self.scorer.ComputeScore(source,target,self.n,mindiff,np.array([source.shape[1]]));
		return cost[0];

	def findMindiff(self,variable):
		sorted_v 	= np.copy(variable);
		sorted_v.sort(axis=0);
		diff 		= np.abs(sorted_v[1]-sorted_v[0]);
		
		if diff==0: diff=np.array([0.001]);
		
		for i in range(1,len(sorted_v)-1):
			curr_diff=np.abs(sorted_v[i+1]-sorted_v[i]);
			if curr_diff!=0 and curr_diff < diff:
				diff = curr_diff;

		diff = diff if diff>1e-9 else 1e-9
		return diff;
