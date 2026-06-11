import numpy as np
from scipy.optimize import linear_sum_assignment

from src.env.entity import SubTask, Worker
import networkx as nx


def KM(
        subtasks: list[SubTask],
        workers: list[Worker],
        now_time: int
) -> tuple[list[tuple[SubTask, Worker]], float]:
    

    subtasks = subtasks.copy()
    workers = workers.copy()

    if not subtasks or not workers:
        return [], 0.

    num_s = len(subtasks)
    num_w = len(workers)

    
    
    N = num_s + num_w

    
    
    
    
    cost = np.zeros((num_s, N), dtype=np.float64)

    LARGE = 1e9

    cost[:, :] = LARGE
    
    cost[:, num_w:] = 0.

    
    for i, subtask in enumerate(subtasks):
        for j, worker in enumerate(workers):
            if worker.constraints_ok(subtask, now_time):
                cost[i, j] = -worker.matching_utility(subtask, now_time)

    
    row_ind, col_ind = linear_sum_assignment(cost)

    sum_utility = 0.

    matches = []
    for si, cj in zip(row_ind, col_ind):
        
        if cj < num_w and cost[si, cj] < 0:
            subtask = subtasks[si]
            worker = workers[cj]

            
            if not worker.constraints_ok(subtask, now_time):
                raise RuntimeError("constraints_ok failed")

            matches.append((subtask, worker))

            sum_utility += worker.matching_utility(subtask, now_time)



    return matches, sum_utility


def networkx(
        subtasks: list[SubTask],
        workers: list[Worker],
        now_time: int
) -> tuple[list[tuple[SubTask, Worker]], float]:
    


    G = nx.Graph()

    
    for si, subtask in enumerate(subtasks):
        G.add_node(f"s{si}", bipartite=0, obj=subtask)

    for wj, worker in enumerate(workers):
        G.add_node(f"w{wj}", bipartite=1, obj=worker)

    
    for si, subtask in enumerate(subtasks):
        for wj, worker in enumerate(workers):
            if worker.constraints_ok(subtask, now_time):
                u = worker.matching_utility(subtask, now_time)
                if u > 0:  
                    G.add_edge(f"s{si}", f"w{wj}", weight=u)

    matching = nx.max_weight_matching(
        G,
        maxcardinality=False  
    )

    matches = []
    sum_utility = 0.
    for u, v in matching:
        if u.startswith("s"):
            subtask = G.nodes[u]["obj"]
            worker = G.nodes[v]["obj"]
        else:
            subtask = G.nodes[v]["obj"]
            worker = G.nodes[u]["obj"]
        matches.append((subtask, worker))
        sum_utility += worker.matching_utility(subtask, now_time)

    return matches, sum_utility
