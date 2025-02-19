import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 读取合并后的层次图数据
with open("./data/schema_graph.json", "r") as f:
    graph_data = json.load(f)

# 解析 L0、L1、L2、L3 节点
node_list = graph_data["L0"] + graph_data["L1"] + graph_data["L2"]
node_mapping = graph_data["node_mapping"]

# 构建边索引
edges = torch.tensor(
    [[node_mapping[src], node_mapping[dst]] for src, dst in graph_data["edges"]],
    dtype=torch.long
).t().contiguous()

num_nodes = len(node_list)
x = torch.rand((num_nodes, 16))  # 随机初始化节点特征

# 构建 PyG 图数据
data = Data(x=x, edge_index=edges).to(device)


# 定义 Hierarchical GNN（HGNN）
class HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGNN, self).__init__()
        
        # L0 → L1
        self.gnn_l0_l1 = GCNConv(in_channels, hidden_channels)
        # L1 → L2
        self.gnn_l1_l2 = GCNConv(hidden_channels, out_channels)

        # L2 → L1
        self.gnn_l2_l1 = GCNConv(out_channels, hidden_channels)
        # L1 → L0
        self.gnn_l1_l0 = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        # 自底向上的传播（Bottom-up）
        x_l1 = F.relu(self.gnn_l0_l1(x, edge_index))  # L0 → L1
        x_l2 = F.relu(self.gnn_l1_l2(x_l1, edge_index))  # L1 → L2
        
        # 自顶向下的传播（Top-down）
        x_l1_td = F.relu(self.gnn_l2_l1(x_l2, edge_index)) + x_l1  # L2 → L1
        x_out = F.relu(self.gnn_l1_l0(x_l1_td, edge_index)) + x  # L1 → L0

        return x_out


# 初始化模型和优化器
model = HGNN(16, 32, 16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # 向节点特征添加高斯噪声
    noisy_x = data.x + torch.randn_like(data.x) * 0.1
    
    out = model(noisy_x, data.edge_index)
    loss = F.mse_loss(out, data.x)  # 自监督训练
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 保存训练好的模型
torch.save(model.state_dict(), "./models/hgnn_model.pth")
print("✅ HGNN 模型训练完成并已保存！")
