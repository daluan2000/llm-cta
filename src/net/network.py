import torch
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

from src.config.config import ModelConfig, SystemConfig


class SetEncoder(nn.Module):
    

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pooling: str = "mean"   
    ):
        
        super().__init__()

        assert pooling in ["sum", "mean", "max"]
        self.pooling = pooling
        self.output_dim = output_dim

        
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        
        if x.numel() == 0 or x.shape[0] == 0:
            
            return torch.zeros(self.output_dim, dtype=torch.float32, device=x.device)

        
        h = self.phi(x)  

        
        if self.pooling == "sum":
            pooled = h.sum(dim=0)
        elif self.pooling == "mean":
            pooled = h.mean(dim=0)
        else:  
            pooled, _ = h.max(dim=0)

        
        z = self.rho(pooled)  
        return z


def encode_skill_set(skill_set: set[int], device: str = 'cpu') -> torch.Tensor:
    
    from src.config.config import ModelConfig

    if len(skill_set) == 0:
        return torch.zeros(ModelConfig.skill_encoder_output_dim, device=device)

    multi_hot = torch.zeros(SystemConfig.skill_nums, dtype=torch.float32, device=device)
    for s in skill_set:
        if 0 <= s < SystemConfig.skill_nums:
            multi_hot[s] = 1.0

    
    projection_matrix = ModelConfig.skill_projection_matrix.to(device)
    return torch.matmul(multi_hot, projection_matrix)


class GAT(nn.Module):
    

    def __init__(self, in_dim, hidden_dim, out_dim, heads):
        super().__init__()

        self.gat1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            concat=True
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False
        )


    def forward(self, x, edge_index):
        
        x = self.gat1(x, edge_index)
        x = F.gelu(x)
        x = self.gat2(x, edge_index)
        return x  

class PolicyNet(nn.Module):
    

    def __init__(self, act_dim: int, hidden: int):
        super().__init__()

        
        self.gat = GAT(
            in_dim=ModelConfig.subtask_dim,
            hidden_dim=ModelConfig.subtask_gat_hidden_dim,
            out_dim=ModelConfig.subtask_gat_output_dim,
            heads=ModelConfig.subtask_gat_heads
        )

        
        mlp_input_dim = (ModelConfig.observation_custom_dim + ModelConfig.task_dim +
                        ModelConfig.subtask_dim + ModelConfig.worker_dim * 2 +
                        ModelConfig.subtask_gat_output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.action_head = nn.Linear(hidden, act_dim)

    def _strip_padding(self, items: torch.Tensor) -> torch.Tensor:
        
        if items.numel() == 0:
            return items
        if items.dim() != 2:
            raise ValueError("items must be 2D tensor for padding strip")
        if torch.isnan(items).any():
            valid_mask = ~torch.isnan(items).any(dim=-1)
            return items[valid_mask]
        return items

    def forward(self, x, edge_index, graph_subtasks, graph_index, workers_enc):
        
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if workers_enc.dim() == 1:
            workers_enc = workers_enc.unsqueeze(0)
        if graph_index.dim() == 0:
            graph_index = graph_index.unsqueeze(0)
        if edge_index.dim() == 2:
            edge_index = edge_index.unsqueeze(0)

        

        
        batch_size = graph_index.shape[0]

        
        if workers_enc.dim() == 1:
            workers_enc = workers_enc.unsqueeze(0).expand(batch_size, -1)

        
        subtasks_enc_list = []
        subtask_idx_map = []  
        graph_nodes_list = []
        node_offset = 0

        for i, st in enumerate(graph_subtasks if isinstance(graph_subtasks, list) else [graph_subtasks]):
            subtask_idx_map.append(node_offset)
            if st is not None and not isinstance(st, list) and st.numel() > 0:
                subtasks_enc_list.append(st)
                graph_nodes_list.append(st)
                node_offset += st.shape[0]
            else:
                subtasks_enc_list.append(None)

        
        if graph_nodes_list:
            graph_nodes = torch.cat(graph_nodes_list, dim=0)
        else:
            graph_nodes = torch.empty((0, ModelConfig.subtask_dim), dtype=torch.float32, device=device)

        
        edge_index_list = []
        graph_index_offsets = []

        for i, st in enumerate(graph_subtasks if isinstance(graph_subtasks, list) else [graph_subtasks]):
            if st is not None and not isinstance(st, list) and st.numel() > 0:
                node_count = st.shape[0]
                
                ei_i = edge_index[i]
                if ei_i.numel() > 0 and node_count > 0:
                    edge_mask = (
                        (ei_i[0] >= 0) & (ei_i[1] >= 0) &
                        (ei_i[0] < node_count) & (ei_i[1] < node_count)
                    )
                    ei_i = ei_i[:, edge_mask]
                    if ei_i.numel() > 0:
                        edge_index_list.append(ei_i + subtask_idx_map[i])

                graph_index_offsets.append(graph_index[i] + subtask_idx_map[i])
            else:
                graph_index_offsets.append(torch.tensor(0, device=device))

        
        if edge_index_list:
            edge_index_combined = torch.cat(edge_index_list, dim=1)
        else:
            edge_index_combined = torch.empty((2, 0), dtype=torch.long, device=device)

        
        graph_index_offset = torch.stack(graph_index_offsets, dim=0)

        
        if graph_nodes.shape[0] == 0:
            gat_rep = torch.zeros(batch_size, ModelConfig.subtask_gat_output_dim, device=device)
        else:
            gat_rep = self.gat(graph_nodes, edge_index_combined)[graph_index_offset]

        
        subtask_enc_list = []
        for i, enc in enumerate(subtasks_enc_list):
            if enc is not None and enc.shape[0] > 0:
                local_idx = graph_index[i].item()
                if local_idx < enc.shape[0]:
                    subtask_enc_list.append(enc[local_idx])
                else:
                    subtask_enc_list.append(torch.zeros(ModelConfig.subtask_dim, device=device))
            else:
                subtask_enc_list.append(torch.zeros(ModelConfig.subtask_dim, device=device))

        subtask_enc = torch.stack(subtask_enc_list, dim=0)

        
        y = torch.cat([x, subtask_enc, workers_enc, gat_rep], dim=-1)
        return self.action_head(self.mlp(y))


class ValueNet(nn.Module):
    

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, customs, workers_enc, tasks_enc, subtasks_enc):
        
        x = torch.cat([customs, workers_enc, tasks_enc, subtasks_enc], dim=-1)
        return self.mlp(x)


class CreditNet(nn.Module):
    

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, contrib_features: torch.Tensor) -> torch.Tensor:
        
        return self.mlp(contrib_features)
