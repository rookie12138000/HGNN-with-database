import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取合并后的层次图数据
with open("./data/schema_graph.json", "r") as f:
    graph_data = json.load(f)

# 构建节点映射
node_list = graph_data["L0"] + graph_data["L1"] + graph_data["L2"]
node_mapping = graph_data["node_mapping"]
# 构建边列表
edges = torch.tensor([[node_mapping[src], node_mapping[dst]] for src, dst in graph_data["edges"]], dtype=torch.long)
edge_index = edges.t().contiguous()

num_nodes = len(node_list)
x = torch.rand((num_nodes, 16))  # 随机初始化节点特征

# 创建 PyG 图数据
data = Data(x=x, edge_index=edge_index).to(device)

# 定义简单的 HGNN 模型
class HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGNN, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型和优化器
model = HGNN(16, 32, 16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    loss = F.mse_loss(out, data.x)  # 自监督训练
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 保存训练好的模型
torch.save(model.state_dict(), "./models/hgnn_model.pth")
print("✅ HGNN 模型训练完成并已保存！")
