import scipy.ndimage
import skimage.morphology
import os
from PIL import Image
import cv2
import math
import numpy
import numpy as np 
from multiprocessing import Pool
import subprocess
import sys
from math import sqrt
import pickle

###-----utils-----####
def drawgraph_inorder(all_list, filename):
    """
    graph: list, a list of vertices list
    [(x1,y1),(x2,y2)...]
    """
    img = np.ones((281,500,3), dtype=np.uint8)*255

    # for n, v in graph:
    #     for nei in v:
    #         p1 = (int(n[1]), int(n[0]))
    #         p2 = (int(nei[1]), int(nei[0]))

    #         img = cv2.line(img, p1, p2, (0,0,0),1)

    # for n, v in graph:
    #     p1 = (int(n[1]), int(n[0]))
    #     img = cv2.circle(img, p1, 0, (0,0,255),2)
    for id,graph in enumerate(all_list):
        # img = np.ones((281,500,3), dtype=np.uint8)*255
        for i in range(len(graph)-1):
            p1 = int(graph[i][1]),int(graph[i][0])
            p2 = int(graph[i+1][1]),int(graph[i+1][0])
            img = cv2.line(img, p1, p2, (0,0,0),1)
            img = cv2.circle(img, p1, 0, (0,0,255),2)
            cv2.imwrite('tmp_%i.jpg'%id, img)

def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d

def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results

def add_edge(src, dst,edges,vertices):
    if (src, dst) in edges or (dst, src) in edges:
        return
    elif src == dst:
        return
    l2_distance = distance(vertices[src],vertices[dst])
    edges.add((src, dst,l2_distance))
    
###----build the graph----###

def build_neighbours(vertices,edges):
    neighbors = {}
    vertex = vertices
    for edge in edges:
        nk1 = (vertex[edge[0]][1],vertex[edge[0]][0])
        nk2 = (vertex[edge[1]][1],vertex[edge[1]][0])
        if nk1 != nk2:
            if nk1 in neighbors:
                if nk2 in neighbors[nk1]:
                    pass
                else:
                    neighbors[nk1].append(nk2)
            else:
                neighbors[nk1] = [nk2]
            if  nk2 in neighbors:
                if nk1 in neighbors[nk2]:
                    pass 
                else:
                    neighbors[nk2].append(nk1)
            else:
                neighbors[nk2] = [nk1]
    return neighbors
def build_graph(img,thin=True):
    """
    Convert the skeleton image to graph structure. Returns with vertices, weighted edges,neighbours, 
    img: 0-1 binary image, single channel
    """
    if thin:
        img = skimage.morphology.thin(img)
    img = img.astype('uint8')
    vertices = []
    edges = set()
    point_to_neighbors = {}
    q = []
    
    while True:
        if len(q) > 0:
            lastid, i, j = q.pop()
            path = [vertices[lastid], (i, j)]
            if img[i, j] == 0:
                continue
            point_to_neighbors[(i, j)].remove(lastid)
            if len(point_to_neighbors[(i, j)]) == 0:
                del point_to_neighbors[(i, j)]
        else:
            w = numpy.where(img > 0)
            if len(w[0]) == 0:
                break
            i, j = w[0][0], w[1][0]
            lastid = len(vertices)
            vertices.append((i, j))
            path = [(i, j)]

        while True:
            img[i, j] = 0
            neighbors = []
            for oi in [-1, 0, 1]:
                for oj in [-1, 0, 1]:
                    ni = i + oi
                    nj = j + oj
                    if ni >= 0 and ni < img.shape[0] and nj >= 0 and nj < img.shape[1] and img[ni, nj] > 0:
                        neighbors.append((ni, nj))
            if len(neighbors) == 1 and (i, j) not in point_to_neighbors:
                ni, nj = neighbors[0]
                path.append((ni, nj))
                i, j = ni, nj
            else:
                if len(path) > 1:
                    path = rdp(path, 2)
                    if len(path) > 2:
                        for point in path[1:-1]:
                            curid = len(vertices)
                            vertices.append(point)
                            add_edge(lastid, curid,edges,vertices)
                            lastid = curid
                    neighbor_count = len(neighbors) + len(point_to_neighbors.get((i, j), []))
                    if neighbor_count == 0 or neighbor_count >= 2:
                        curid = len(vertices)
                        vertices.append(path[-1])
                        add_edge(lastid, curid,edges,vertices)
                        lastid = curid
                for ni, nj in neighbors:
                    if (ni, nj) not in point_to_neighbors:
                        point_to_neighbors[(ni, nj)] = set()
                    point_to_neighbors[(ni, nj)].add(lastid)
                    q.append((lastid, ni, nj))
                for neighborid in point_to_neighbors.get((i, j), []):
                    add_edge(neighborid, lastid,edges,vertices)
                break
            
    vertex = vertices
    neighbors = build_neighbours(vertex,edges)

    return vertices,edges,neighbors

class UnionFind(object): # taken from https://leetcode.cn/problems/min-cost-to-connect-all-points/solution/-by-larrychen__-ij0v/
    def __init__(self, n):
        self._count = n
        self._parent = [i for i in range(n)] # initialize with all vertices' indices
        self._size = [1] * n

    def union(self, p, q):
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return False

        # 小树连接到大树上面
        if self._size[root_p] > self._size[root_q]:
            self._parent[root_q] = root_p
            self._size[root_p] += self._size[root_q]
        else:
            self._parent[root_p] = root_q
            self._size[root_q] += self._size[root_p]
        
        self._count -= 1
        return True
        
    def find(self, x):
        while x != self._parent[x]:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def get_count(self):
        return self._count

    def connected(self, p, q):
        return self.find(p) == self.find(q)
    
    
###---build max kruskal---###
def build_kruskal(vertices,edges,neighbours):
    '''
    build the  kruskal tree
    vertices: list,[(x1,y1),(x2,y2)...]
    edges: set, {(vertex1,vertex2,distance),....} the vertex1 is the index in vertices.
    neighbours: dict, { (x1,y1):[(xi,yi),(xj,yj)....] }
        
    return: vertices, new_edges, new_neighbours
    vertices is not changed
    '''
    num_vertices = len(vertices)
    edges = list(edges)
    edges = sorted(edges,key=lambda x:x[-1],reverse=False) # sort the edges by distance in ascending order
    new_edges = set()
    new_neighbours = {}
    uf = UnionFind(num_vertices)
    for v1,v2,di in edges:        # determin wether there is a loop in the tree
        if uf.connected(v1,v2): # 2 vertices has been connected via other vertices
            continue
        else: # build the max kruskal tree
            uf.union(v1,v2)
            new_edges.add((v1,v2,di))
    new_neighbours = build_neighbours(vertices,new_edges)
    return vertices, new_edges, new_neighbours
###-----DFS-----#
def DFS(curr_vet,neighbours,visited=set(),curr_path=[],all_paths=[]):
    '''
    curr_vet: a vertex, (x,y)
    vertices: list,[(y1,x1),(y2,x2)...]
    neighbours: dict, { (x1,y1):[(xi,yi),(xj,yj)....] }

    return: all paths start from curr_vet
    '''
    res = []
    visited.add(curr_vet)
    is_end = True
    curr_neighbours = neighbours[curr_vet]
    for neibour in curr_neighbours:
        if neibour not in visited:
            is_end = False
            # break
    if is_end:
        curr_path.append(curr_vet)
        all_paths.append(curr_path.copy())
        curr_path.pop()
        return all_paths
    curr_path.append(curr_vet)
    for vet in curr_neighbours:
        if vet in visited:
            # curr_path.remove(vet)
            continue
        else:
            all_paths= DFS(vet,neighbours,visited,curr_path,all_paths)
            # print(all_paths)
    res = all_paths
    curr_path.pop()
    return res

