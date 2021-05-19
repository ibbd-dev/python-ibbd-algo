# 最短距离算法
import networkx as nx

debug = False
start_node = ('start', -1)    # 初始节点
end_node = ('end', -1)        # 终止节点


def fmt_edges(points, max_score=1.):
    """将节点得分列表格式化成距离矩阵
    :param points list[[left, right, score]]
    :return edges [(node_id1, node_id2, score)]
    :return nodes [(left, right)]
    """
    points_dict = dict()
    for i, j, score in points:
        # 默认值为虚拟节点
        points_dict.setdefault(i, [(-1, max_score)]).append((j, max_score-score))
    
    edges, last_nodes = [], [start_node]
    nodes = [start_node]
    for left, points_score in points_dict.items():
        curr_nodes = []
        for right, score in points_score:
            node = (left, right)
            curr_nodes.append(node)
            edges += init_edges(last_nodes, node, score)

        nodes += curr_nodes
        last_nodes = curr_nodes
    
    # 终止节点
    nodes.append(end_node)
    if debug:
        print('edges:', [edge[:2] for edge in edges if edge[0] != start_node])

    edges += init_edges(last_nodes, end_node, 0.)
    node_keys = {val: key for key, val in enumerate(nodes)}
    edges = [(node_keys[f_node], node_keys[t_node], score) for f_node, t_node, score in edges]
    return edges, nodes


def init_edges(last_nodes, point, score):
    """"""
    edges = []
    for last in last_nodes:
        if last[1] >= 0 and point[1] >= 0 and last[1] >= point[1]:
            continue
        edges.append((last, point, score))

    return edges


def shortest_distance(edges, nodes, target=None, source=0):
    """最短距离算法
    :return path list[(left, right)]
    """
    if target is None:
        target = len(nodes) - 1

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    path = nx.dijkstra_path(G, source=source, target=target)
    if debug:
        print('shortest path: ', path)
    path = [p for p in path if p not in set((source, target))]
    path = [nodes[p] for p in path]
    return path


if __name__ == '__main__':
    debug = True
    def test(points):
        edges, nodes = fmt_edges(points)
        path = shortest_distance(edges, nodes)
        print(path)
        print("-------")

    points = [[0, 0, 0.5]]
    test(points)
    points = [[0, 0, 0.7], [0, 1, 0.1], [1, 0, 0.2], [1, 1, 0.6]]
    test(points)
    points = [[0, 0, 0.7], [0, 1, 0.1], [1, 0, 0.8], [1, 1, 0.6]]
    test(points)
    points = [[0, 0, 0.7], [1, 0, 0.1], [1, 1, 0.2], [2, 1, 0.8]]
    test(points)