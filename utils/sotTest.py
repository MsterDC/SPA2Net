# -*- coding: UTF-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def _reformat(adj_dict):
    nodes = []
    node_list = []
    for k, v in adj_dict.items():
        nodes.append(k)
        for _v in v:
            node_list.append((k, (_v[0], _v[1]), _v[-1]))

    return nodes, node_list


def lsc(f_phi):
    n, c_nl, h, w = f_phi.size()
    c_nl = f_phi.size(1)
    f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
    f_phi_normed = f_phi / (torch.norm(f_phi, dim=2, keepdim=True) + 1e-10)

    # first order（一阶HC）
    non_local_cos = F.relu(
        torch.matmul(f_phi_normed, f_phi_normed.transpose(1, 2)))  # (1,196,196) = (1,c,196)×(1,196,c)
    return non_local_cos


def hsc(f_phi, fo_th=0.1, so_th=0.1, order=2):
    """
    Calculate affinity matrix and update feature.
    :param f_phi: feature map, size=(1,c,14,14)
    :param fo_th: 一阶阈值
    :param so_th: 二阶阈值
    :param order: 指定阶数(2阶)
    :return: SCM
    """
    n, c_nl, h, w = f_phi.size()
    c_nl = f_phi.size(1)
    f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
    f_phi_normed = f_phi / (torch.norm(f_phi, dim=2, keepdim=True) + 1e-10)

    # first order（一阶HC）
    non_local_cos = F.relu(
        torch.matmul(f_phi_normed, f_phi_normed.transpose(1, 2)))  # (1,196,196) = (1,c,196)×(1,196,c)
    # print("non_local_cos:")
    # print(non_local_cos)
    non_local_cos[non_local_cos < fo_th] = 0
    non_local_cos_fo = non_local_cos.clone()
    non_local_cos_fo = non_local_cos_fo / (torch.sum(non_local_cos_fo, dim=1, keepdim=True) + 1e-5)

    # high order（二阶HC）
    base_th = 1. / (h * w)
    so_th = base_th * so_th
    # 对角线清零
    non_local_cos[:, torch.arange(h * w), torch.arange(w * h)] = 0
    # 归一化
    non_local_cos = non_local_cos / (torch.sum(non_local_cos, dim=1, keepdim=True) + 1e-5)
    non_local_cos_ho = non_local_cos.clone()
    for _ in range(order - 1):
        # 矩阵自乘
        non_local_cos_ho = torch.matmul(non_local_cos_ho, non_local_cos)
        # 归一化
        non_local_cos_ho = non_local_cos_ho / (torch.sum(non_local_cos_ho, dim=1, keepdim=True) + 1e-10)
    # 阈值过滤
    non_local_cos_ho[non_local_cos_ho < so_th] = 0

    return non_local_cos_fo, non_local_cos_ho


def gen_adjacency_list(features):
    """
    Generating graph based on features obtained from backbone.
    :param features: input features, shape:(b,c,h,w).
    :return: graph with weight, which saved as dict.
    eg: graph = {
        '(x,y)': [(x_1,y_1,w_1),...], adjacency list
        ...
    }
    """
    (n, c_nl, h, w) = features.shape
    graph = {}
    # assert h == 14 and w == 14, "wrong size of feature"
    for i in range(h):
        for j in range(w):
            if i == 0:  # top
                if j == 0:  # left top
                    graph[(i, j)] = [(i, j + 1), (i + 1, j)]
                elif j == w - 1:  # right top
                    graph[(i, j)] = [(i, j - 1), (i + 1, j)]
                else:
                    graph[(i, j)] = [(i, j - 1), (i + 1, j), (i, j + 1)]
            elif i == h - 1:  # bottom
                if j == 0:  # left bottom
                    graph[(i, j)] = [(i - 1, j), (i, j + 1)]
                elif j == w - 1:  # right bottom
                    graph[(i, j)] = [(i, j - 1), (i - 1, j)]
                else:
                    graph[(i, j)] = [(i, j - 1), (i - 1, j), (i, j + 1)]
            elif (i != 0 and i != h - 1) and j == 0:  # left
                graph[(i, j)] = [(i - 1, j), (i, j + 1), (i + 1, j)]
            elif (i != 0 and i != h - 1) and j == w - 1:  # right
                graph[(i, j)] = [(i - 1, j), (i, j - 1), (i + 1, j)]
            else:  # inner
                graph[(i, j)] = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    # generate weight
    for startNode, list_adj in graph.items():
        for index, adjNode in enumerate(list_adj):
            x1, x2 = features[:, :, startNode[0], startNode[1]], \
                     features[:, :, adjNode[0], adjNode[1]]
            cos_sim = torch.cosine_similarity(x1, x2, dim=1)
            # w = 1 - cos_sim
            w = cos_sim
            graph[startNode][index] = graph[startNode][index] + (w,)
    return graph


def reformat_to_map(graph, features):
    """
    Transform graph<dict> to maps<list>
    :param graph: graph with weight
    :param features: features obtained from backbone
    :return: maps<list>
    """
    (n, c_nl, h, w) = features.shape
    # M: float = float('inf')
    M: float = 2.0
    maps = []
    for i in range(h):
        for j in range(w):
            adj_nodes = graph[(i, j)]
            row = [M] * h * w
            row[w * i + j] = 0
            for n_i, node in enumerate(adj_nodes):
                (x, y), dists = (node[0], node[1]), node[-1].float()
                idx = w * x + y
                row[idx] = dists[0].item()
            maps.append(row)
    return maps


class Graph(object):
    def __init__(self, length: int, matrix: [], vertex: []):
        """
        :param length: 大小
        :param matrix: 邻接矩阵
        :param vertex: 顶点数组
        """
        self.dis = matrix
        self.vertex = vertex
        self.len = length

    def floyd(self):
        for k in range(self.len):
            for i in range(self.len):
                for j in range(self.len):
                    # pass
                    self.dis[i][j] = min(self.dis[i][j], self.dis[i][k] + self.dis[k][j])

    def show_graph(self):
        result = []
        for k in range(len(self.dis)):
            for i in range(len(self.dis)):
                result.append(self.dis[k][i])
        shape = (1, int(len(result) ** 0.5), int(len(result) ** 0.5))
        res = torch.reshape(torch.from_numpy(np.array(result)), shape)
        res = 1 - res
        res = res / (torch.sum(res, dim=1, keepdim=True) + 1e-10)
        res = F.relu(res)
        return res


def test(features):
    _, _, h, w = features.size()
    vertex: [] = [i for i in range(h * w)]
    print("vertex:", vertex)
    print()
    # graph = gen_adjacency_list(features)
    # matrix = reformat_to_map(graph, features)
    # print(fo.shape)
    # matrix = (1-fo).squeeze(dim=0).numpy().tolist()
    # print("ori matrix:")
    # for d in matrix:
    #     print(d)
    graph = gen_adjacency_list(features)
    print("adjacency_list:")
    for k, v in graph.items():
        print(k, "=>", v)
    print()
    nodes, node_list = _reformat(graph)
    print(nodes)
    print("reformat: ")
    for n in node_list:
        print(n)
    print()
    matrix = reformat_to_map(graph, features)
    print("matrix:")
    for m in matrix:
        print(m)
    print()
    g = Graph(len(vertex), matrix, vertex)
    g.floyd()
    scm = g.show_graph()
    scm[:, torch.arange(h * w), torch.arange(h * w)] = 0
    scm = scm / (torch.sum(scm, dim=1, keepdim=True) + 1e-5)
    print("spm: ")
    for s in scm:
        print(s)
    return scm


def main(features):
    _, _, h, w = features.size()
    vertex: [] = [i for i in range(h * w)]
    graph = gen_adjacency_list(features)
    matrix = reformat_to_map(graph, features)
    g = Graph(len(vertex), matrix, vertex)
    g.floyd()
    sot_sc = g.show_graph()
    sot_sc[:, torch.arange(h * w), torch.arange(h * w)] = 0
    sot_sc = sot_sc / (torch.sum(sot_sc, dim=1, keepdim=True) + 1e-5)
    # scm = sot_sc.clone()
    # scm = torch.matmul(scm, sot_sc)
    # scm = scm / (torch.sum(scm, dim=1, keepdim=True) + 1e-10)
    return sot_sc


# if __name__ == '__main__':
#     features = np.random.random([1, 256, 3, 3])
#     features = torch.from_numpy(features)
#     fo, so = hsc(features)
#     print("first order: ")
#     for f in fo:
#         print(f)
#     print()
#     spm = test(features)


################################## Another method #######################################
""" 
Created on Thu Jul 13 14:56:37 2017 

@author: linzr 
"""

## 表示无穷大
INF_val = -1


class Floyd_Path():
    def __init__(self, node, node_map, path_map):
        self.node = node
        self.node_map = node_map
        self.node_length = len(node_map)
        self.path_map = path_map
        self._init_Floyd()

    def __call__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        return self._format_path()

    def _init_Floyd(self):
        for k in range(self.node_length):
            for i in range(self.node_length):
                for j in range(self.node_length):
                    tmp = self.node_map[i][k] * self.node_map[k][j]
                    if self.node_map[i][j] < tmp:
                        self.node_map[i][j] = tmp
                        self.path_map[i][j] = self.path_map[i][k]
                    # 目前存储的是图中各点之间的距离
                    # 如果由加法换成乘法，即计算相似度的乘积，需要先将距离转换为相似度相乘
                    # s_i_k, s_k_j = 1 - self.node_map[i][k], 1 - self.node_map[k][j]
                    # product_s = float(s_i_k * s_k_j)
                    # if (1-self.node_map[i][j]) < product_s:
                    #     self.node_map[i][j] = 1 - product_s
                    #     self.path_map[i][j] = self.path_map[i][k]

        # print('_init_Floyd is end')

    def _format_path(self):
        node_list = []
        temp_node = self.from_node
        obj_node = self.to_node
        print("===", self.node_map[temp_node][obj_node])
        node_list.append(self.node[temp_node])
        while True:
            node_list.append(self.node[self.path_map[temp_node][obj_node]])
            temp_node = self.path_map[temp_node][obj_node]
            if temp_node == obj_node:
                break

        return node_list


def set_node_map(node_map, node, node_list, path_map):
    for i in range(len(node)):
        ## 对角线为0
        node_map[i][i] = 0
    for x, y, val in node_list:
        node_map[node.index(x)][node.index(y)] = node_map[node.index(y)][node.index(x)] = val
        path_map[node.index(x)][node.index(y)] = node.index(y)
        path_map[node.index(y)][node.index(x)] = node.index(x)


def _test(features):
    _, _, h, w = features.size()
    graph = gen_adjacency_list(features)
    # print("adjacency_list:")
    # for k, v in graph.items():
    #     print(k, "=>", v)
    # print()
    node, node_list = _reformat(graph)
    # print(node)
    print("reformat: ")
    for n in node_list:
        print(n)
    print()

    ## node_map[i][j] 存储i到j的最短距离
    node_map = [[INF_val for val in range(len(node))] for val in range(len(node))]
    ## path_map[i][j]=j 表示i到j的最短路径是经过顶点j
    path_map = [[0 for val in range(len(node))] for val in range(len(node))]

    ## set node_map
    set_node_map(node_map, node, node_list, path_map)

    ## select one node to obj node, e.g. A --> D(node[0] --> node[3])
    Floydpath = Floyd_Path(node, node_map, path_map)
    for i in range(len(node)):
        for j in range(len(node)):
            print("from:", node[i], "to:", node[j], end="")
            path = Floydpath(i, j)
            print(path)
            print()


def max_hsc_ori(non_local_cos):
    max_scm = []
    # print("non_local_cos:")
    # print(non_local_cos)
    for i in range(len(non_local_cos[0])):
        for j in range(len(non_local_cos[0][0])):
            max_scm.append(max(non_local_cos[0][j].mul(non_local_cos[0][i])))
    shape = (1, int(len(max_scm) ** 0.5), int(len(max_scm) ** 0.5))
    scm = torch.reshape(torch.from_numpy(np.array(max_scm)), shape)
    scm = scm / (torch.sum(scm, dim=1, keepdim=True) + 1e-5)
    return scm


def max_hsc(f_phi, fo_th=0.2, so_th=1):
    n, c_nl, h, w = f_phi.size()
    c_nl = f_phi.size(1)
    f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
    f_phi_normed = f_phi / (torch.norm(f_phi, dim=2, keepdim=True) + 1e-10)

    # first order
    non_local_cos = F.relu(torch.matmul(f_phi_normed, f_phi_normed.transpose(1, 2)))
    non_local_cos[non_local_cos < fo_th] = 0
    non_local_cos_fo = non_local_cos.clone()
    non_local_cos_fo = non_local_cos_fo / (torch.sum(non_local_cos_fo, dim=1, keepdim=True) + 1e-5)

    # max high order
    max_scm = []
    # max_scm_ = []

    non_local_cos[:, torch.arange(h * w), torch.arange(w * h)] = 0
    # non_local_cos = non_local_cos / (torch.sum(non_local_cos, dim=1, keepdim=True) + 1e-5)
    base_th = 1. / (h * w)
    so_th = base_th * so_th
    non_local_cos_ho = non_local_cos.clone()
    # print("non_local_cos_ho:", non_local_cos_ho.device)
    # print("non_local_cos:", non_local_cos.device)
    non_local_cos = non_local_cos.cpu()
    non_local_cos_ho = non_local_cos_ho.cpu()
    print("non_local_cos_ho:",'\n',non_local_cos_ho,non_local_cos_ho.shape)
    print("non_local_cos:",'\n',non_local_cos,non_local_cos.shape)

    for i in range(len(non_local_cos[0])):
        for j in range(len(non_local_cos[0][0])):
            max_scm.append(max(non_local_cos_ho[0][j].mul(non_local_cos[0][i])))
    shape = (1, int(len(max_scm) ** 0.5), int(len(max_scm) ** 0.5))
    scm = torch.reshape(torch.tensor(max_scm), shape)
    scm = scm / (torch.sum(scm, dim=1, keepdim=True) + 1e-10)
    scm[scm < so_th] = 0

    return non_local_cos_fo, scm


def inter(features):
    non_local_cos = lsc(features)
    scm = max_hsc(non_local_cos)
    return scm


if __name__ == "__main__":
    features = np.random.random([1, 256, 4, 4])
    features = torch.from_numpy(features)
    fo,so = max_hsc(features)
    # print(fo,'\n',so)
    # non_local_cos = lsc(features)
    # max_hsc(non_local_cos)
    # fo, so = hsc(features)
    # print("fo:", fo)
    # _test(features)
