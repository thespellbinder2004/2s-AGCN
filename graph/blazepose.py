import numpy as np

num_node = 33

self_link = [(i, i) for i in range(num_node)]

inward = [
    (1, 0), (2, 0), (3, 1), (4, 2),
    (5, 0), (6, 0), (7, 5), (9, 7), (8, 6), (10, 8),
    (6, 5),
    (11, 5), (12, 6), (12, 11),
    (13, 11), (15, 13),
    (14, 12), (16, 14),
    (23, 11), (24, 12), (24, 23),
    (25, 23), (26, 24),
    (27, 25), (28, 26),
    (29, 27), (30, 28),
    (31, 29), (32, 30)
]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            return get_spatial_graph(num_node, self_link, inward, outward)
        raise ValueError(f"Unsupported labeling mode: {labeling_mode}")