import torch
import torch.nn.functional as F
import numpy as np

# A Python program for Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix representation of the graph

import sys # Library for INT_MAX

class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]for row in range(vertices)]
        # print(self.graph)

    # A utility function to print the constructed MST stored in parent[]
    def printMST(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][ parent[i] ])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):
        # Initialize min value
        min = float("inf")
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def get_neiborlist(self, f_phi):
        n, c_nl, h, w = f_phi.size()
        f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
        f_phi_normed = f_phi / (torch.norm(f_phi, dim=2, keepdim=True) + 1e-10)
        sim_ft = F.relu(torch.matmul(f_phi_normed, f_phi_normed.transpose(1, 2)))
        sim_ft = 1.0 - sim_ft  # sim transform to distance => n * wh * wh
        sim_ft[:, torch.arange(h * w), torch.arange(w * h)] = 0
        sim_ft = sim_ft / (torch.sum(sim_ft, dim=1, keepdim=True) + 1e-5)
        return sim_ft

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):
        # Key values used to pick minimum weight edge in cut
        key = [float("inf")] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if 0 < self.graph[u][v] < key[v] and mstSet[v] == False:
                        key[v] = self.graph[u][v]
                        parent[v] = u
        self.printMST(parent)



# g = Graph(5)
# g.graph = [ [0, 2, 0, 6, 0],
#             [2, 0, 3, 8, 5],
#             [0, 3, 0, 0, 7],
#             [6, 8, 0, 0, 9],
#             [0, 5, 7, 9, 0]]

feature = torch.rand(1,3,3,3)
g = Graph(feature.shape[-1]*feature.shape[-2])
adj_list = g.get_neiborlist(feature)
print("adj_list:", adj_list)
for i in range(adj_list.shape[0]):
    g.graph = adj_list[i].tolist()
    g.primMST()

g.primMST()
