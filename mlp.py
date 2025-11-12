import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import ast
epochs = 20
batch_size = 1
# 设置 Early Stopping 参数
patience = 5  # 允许多少个 epoch 没有改善后停止训练
best_loss = np.inf  # 最初的最好损失设为无穷大
epochs_without_improvement = 0  # 连续没有改善的 epoch 数
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self,inner_dim=64,base_y=22.5):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 1,bias=False)
        )
        self.base_y=base_y
    def forward(self, x):
        x = x.view(-1, 20)
        return self.model(x)+self.base_y

# 从txt文件读取数据
def load_data_from_txt(file_path):
    X = []
    y = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # 使用ast.literal_eval来安全地解析元组形式的字符串
            tuple_data = ast.literal_eval(line)  
            X.append(tuple_data[:-1])  
            y.append(tuple_data[-1])  
    return np.array(X), np.array(y).reshape(-1, 1)  # 返回X和y作为numpy数组
# 加载数据
X, y = load_data_from_txt("data.txt")
# 划分训练集和验证集
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
# 转换为Tensor
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()
# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
# 初始化模型、损失函数、优化器
model = MLP(base_y=22.5)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.002,momentum=0.4)
train_losses = []
val_losses = []
# 训练模型
print("开始训练...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # 保存训练损失
    # 每个 epoch 后评估一次验证集损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
    # 打印每个 epoch 的损失
    val_losses.append(val_loss)  # 保存测试损失
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, val Loss: {val_loss:.4f}')

    # Early Stopping 判断
    if val_loss < best_loss:
        best_loss = val_loss  # 更新最好的损失值
        epochs_without_improvement = 0  # 重置计数器
    else:
        epochs_without_improvement += 1  # 记录连续未改进的 epoch 数
    # 如果超过 patience 次没有改善，则提前停止训练
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}, validation loss did not improve for {patience} epochs.")
        break  # 提前停止训练
# 最终测试
print("\n最终测试结果:")
model.eval()
with torch.no_grad():
    predictions = model(X_val_tensor)
    final_loss = criterion(predictions, y_val_tensor).item()
    # 计算R²分数
    y_val_np = y_val_tensor.cpu().numpy()
    pred_np = predictions.cpu().numpy()
    ss_res = np.sum((y_val_np - pred_np) ** 2)
    ss_tot = np.sum((y_val_np - y_val_np.mean()) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    print(f'Final val Loss (MSE): {final_loss:.4f}')
    print(f'R² Score: {r2_score:.4f}')

# 显示几个预测示例
print("\n预测示例（前5个）:")
print("真实值\t\t预测值")
for i in range(5):
    print(f'{y_val_np[i][0]:.4f}\t\t{pred_np[i][0]:.4f}')
# 绘制训练损失曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
epochs_range = range(1, len(train_losses) + 1)
plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(epochs_range, val_losses, 'r-', label='val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training and val Loss over Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n损失曲线已保存为 'training_loss_curve.png'")