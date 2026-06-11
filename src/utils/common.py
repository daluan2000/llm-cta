import torch


def flatten(nested_list: list):


    result = []
    for item in nested_list:
        if isinstance(item, list):  
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def adj_matrix_to_edge_index(graph_matrix: list[list[int]], device: torch.device | None = None):

    src, dst = [], []
    n = len(graph_matrix)
    for i in range(n):
        for j in range(len(graph_matrix[i])):
            if graph_matrix[i][j] != 0:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index
