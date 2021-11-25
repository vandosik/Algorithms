import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

v_num = 100  # vertices number
e_num = 200  # edges number


def DFS_find_comps(g, n):
    global vis
    if n not in vis:
        vis.append(n)
        for i in g[n]:
            DFS_find_comps(g, i)


# find item, that is not presented in any collection
def find_new(components):
    if len(components) == 0:
        return 0
    keys = components.keys()
    for j in range(0, v_num):
        flag = 1
        for i in keys:
            if j in components[i]:
                flag = 0
        if flag == 1:
            return j
    return -1

def BFS_shortest_path(a, b):

    global g_list
    v_num = len(g_list)

    cur_node = a
    fin_node = b
    queue = []
    way_to_node = [[] * v_num for i in range(v_num)]
    checked = [False for i in range(v_num)]

    queue.append(cur_node)
    checked[cur_node] = True
    way_to_node[cur_node] = [cur_node]

    while len(queue) > 0:
        cur_node = queue.pop(0)  # get first node from queue

        if cur_node == fin_node:
            break

        for neigh_node in g_list[cur_node]:
            if not checked[neigh_node]:
                queue.append(neigh_node)  # add node to queue tail
                checked[neigh_node] = True
                way_to_node[neigh_node] = way_to_node[cur_node] + [neigh_node]  # append way to node

    return way_to_node[fin_node]


if __name__ == '__main__':
    g_mat = np.zeros((v_num, v_num))
    new_edges = 0
    while new_edges < e_num:
        i = random.randint(0, v_num - 1)
        j = random.randint(0, v_num - 1)
        if g_mat[i, j] == 0:
            g_mat[i, j] = 1
            g_mat[j, i] = 1
            new_edges += 1

    print('Matrix creation finished')
    for i in range(0, 3):
        print('(', end='')
        for j in range(0, v_num):
            print(int(g_mat[i][j]), end=''),
        print(')')

    g_list = [[] * v_num for i in range(v_num)]

    for i in range(0, v_num):
        for j in range(0, v_num):
            if g_mat[i, j] == 1:
                g_list[i].append(j)

    print('List creation finished')
    for num, i in enumerate(g_list):
        print(num, ':', i)
        if num > 3:
            break

    G = nx.from_numpy_matrix(g_mat)
    print('Components')
    components = {}  # create dictionary

    end = 0
    i = 0
    while end == 0:
        value = find_new(components)

        if value != -1:
            vis = []
            DFS_find_comps(g_list, value)
            components[i] = vis
            print('Connected component', i, '= ', components[i])
        else:
            end = 1
        i += 1

    i = random.randint(0, v_num - 1)
    j = random.randint(0, v_num - 1)
    print('Shortest (found using network) x path from', i, 'to', j, 'is', nx.bidirectional_shortest_path(G, i, j))
    print('Shortest (manually found) path from', i, 'to', j, 'is', BFS_shortest_path(i, j))

    nx.draw(G, pos=nx.random_layout(G), node_color='red', edge_color='grey', node_size=5)
    plt.show()