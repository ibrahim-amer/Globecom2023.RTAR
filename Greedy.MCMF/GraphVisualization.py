# First networkx library is imported
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt


# Defining a Class
class GraphVisualization:

	def __init__(self):
		
		# visual is a list which stores all
		# the set of edges that constitutes a
		# graph
		self.visual = []
		self.graphs_count = 0
		
	# addEdge function inputs the vertices of an
	# edge and appends it to the visual list
	def addEdge(self, a, b):
		temp = [a, b]
		self.visual.append(temp)
		
	def buildGraphFromAdjMatrix(self, adj, graph_title = ''):
		G = nx.DiGraph()
		for i in range(0, len(adj)):
			for j in range(0, len(adj)):
				G.add_edge(i, j, weight=adj[i][j])
		nx.draw_networkx(G)
		self.graphs_count += 1
		plt.figure(self.graphs_count)
		plt.title(graph_title)

	def showGraphs(self):
		plt.show()
		
	# In visualize function G is an object of
	# class Graph given by networkx G.add_edges_from(visual)
	# creates a graph with a given list
	# nx.draw_networkx(G) - plots the graph
	# plt.show() - displays the graph
	def visualize(self):
		G = nx.Graph()
		G.add_edges_from(self.visual)
		nx.draw_networkx(G)
		plt.show()
    
    

# Driver code
# G = GraphVisualization()
# G.addEdge(0, 2)
# G.addEdge(1, 2)
# G.addEdge(1, 3)
# G.addEdge(5, 3)
# G.addEdge(3, 4)
# G.addEdge(1, 0)
# G.visualize()
