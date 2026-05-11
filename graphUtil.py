from copy import deepcopy

class GraphUtil:

	def causesCycle(self,es,V):
		edge_set=deepcopy(es[0]);
		edge_set.add(es[1]);

		adj_mat=[[0 for y in range(V)] for x in range(V)];
		for e in edge_set:
			parent=e[0];
			child=e[1];
			adj_mat[child][parent]=1;

		flag= self.HasCycle(adj_mat);
		return flag;

	def HasCycle(self,graph):
		V = len(graph);
		visited = [False] * V;
		onStack = [False] * V;
		
		for n in range(V):
			if not visited[n]:
				if self.CycleChecker(graph,n,visited,onStack) == True:
					return True;
		return False;
	
	def CycleChecker(self,graph,node,visited,stack):
		visited[node]=True;
		stack[node]=True;
		
		neighbours=[i for i,x in enumerate(graph[node]) if x !=0];
		
		for neighbour in neighbours:
			if not visited[neighbour]:
				if self.CycleChecker(graph,neighbour,visited,stack):
					return True;
			elif stack[neighbour]==True:
				return True;
				
		stack[node]=False;
		return False;
		
