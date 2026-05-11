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
np.random.seed(13)	#for laplacian noise
from ges.decomposable_score import DecomposableScore
from alg.globe.globe import Globe
import random
import math


class PeriScore(DecomposableScore):
	"""
	Implements  Federated Causal learning score

	"""

	def __init__(self, data, envs,sub_sample=1.0,ln=0.0,cache=True, debug=0):

		if type(data) != np.ndarray:
			raise TypeError("data should be numpy.ndarray, not %s." % type(data))

		super().__init__(data, cache=cache, debug=debug)
		self.envs=envs;
		self.n, self.p = data.shape
		self.pcache={};
		self.ids_ = [i for i in range(len(envs))]
		self.counts= int(math.floor(sub_sample*len(envs)));
		self.lambda_=ln

	def full_score(self,A0):
		score=0;
		for x in range(self.p):
			pa		=	np.nonzero(A0[:,x])[0].tolist()			#find parent ids'
			pa.sort()
			score	+=	self._compute_local_score(x, pa)		#calculate local score, and sum up

		return score
	def _compute_local_score(self, x, pa):
		"""
		Given a node and its parents, return the maximum regret over
		all environments, by finding the regret over each environment
		and choosing the highest regret.
		"""
		score=0;
		pa=list(pa)
		pa.sort()

		#to handle the case where we want to query subset of clients when there are eg > 100 clients
		query_ids = random.sample(self.ids_,self.counts)		
		
		#since our implementation of GES is meant to maximize the score, 
		#we have to re-cast min max(regret) as max min(-regret) , 
		#negation of regret part should be handled within the env[cc].score() 
		#here we only take the min
		regs=[self.envs[cc].score(pa,x) for cc in query_ids]
		score = min(regs)	+ self.sanitize()
		return score

	def sanitize(self):
		if self.lambda_==0.0: return 0;
		x=np.random.laplace(0,self.lambda_)
		return x;
