import torch
import torch.nn.functional as F
import json
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 定义 HGNN 模型
class HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGNN, self).__init__()
        
        # 设计多层图卷积网络
        self.gnn_l0_l1 = GCNConv(in_channels, hidden_channels)
        self.gnn_l1_l2 = GCNConv(hidden_channels, out_channels)
        self.gnn_l2_l1 = GCNConv(out_channels, hidden_channels)
        self.gnn_l1_l0 = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        # 保持与训练代码相同的传播逻辑
        x_l1 = F.relu(self.gnn_l0_l1(x, edge_index))
        x_l2 = F.relu(self.gnn_l1_l2(x_l1, edge_index))
        x_l1_td = F.relu(self.gnn_l2_l1(x_l2, edge_index)) + x_l1
        x_out = F.relu(self.gnn_l1_l0(x_l1_td, edge_index)) + x
        return x_out

# 加载模型
model = HGNN(16, 32, 16)  # 输入特征16，隐藏层32，输出特征16
model.load_state_dict(torch.load("./models/hgnn_model.pth", map_location="cpu"))  # 确保权重加载到 CPU 上
model.eval()
print("✅ HGNN 模型加载成功！")

# 读取图数据
with open("./data/schema_graph.json", "r") as f:
    graph_data = json.load(f)

# 直接通过层级列表生成节点映射（确保包含所有节点）
node_list = graph_data["L0"] + graph_data["L1"] + graph_data["L2"]
node_mapping = {name: idx for idx, name in enumerate(node_list)}

# 构建边索引
edges = [
    [node_mapping[src], node_mapping[dst]]
    for src, dst in graph_data["edges"]
    if src in node_mapping and dst in node_mapping  # 防止意外缺失
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 构建图数据
data = Data(
    x=torch.rand((len(node_list), 16)),  # 随机初始化节点特征
    edge_index=edge_index
)

# 计算相似度
def compute_similarity(node1, node2):
    idx1 = node_mapping[node1]
    idx2 = node_mapping[node2]
    
    # 获取节点嵌入
    emb1, emb2 = model(data.x, data.edge_index)[idx1], model(data.x, data.edge_index)[idx2]
    
    # 计算余弦相似度
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

# 测试
print("students.class_id 和 classes.id 的相似度:", compute_similarity("school.db.students.class_id", "school.db.classes.id"))
print("school.db 和 company.db 的相似度:", compute_similarity("school.db", "company.db"))
print("employees.department_id 和 classes.id 的相似度:", compute_similarity("company.db.employees.department_id", "school.db.classes.id"))
print("teachers.subject 和 employees.position 的相似度:", compute_similarity("school.db.teachers.subject", "company.db.employees.position"))
