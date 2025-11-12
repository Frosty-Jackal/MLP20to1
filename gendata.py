import numpy as np
# 生成数据
X = np.random.randn(30000, 20).astype(np.float32)
# 构造 y：让每个维度都影响结果
# 综合线性项 + 非线性项 + 交互项
y = (
    1.2 * np.sin(X[:, 0] * X[:, 1])
    + 0.8 * np.cos(X[:, 2] ** 2 - X[:, 3])
    + 0.5 * np.sin(X[:, 4] + X[:, 5] * X[:, 6])
    + 0.7 * np.cos(X[:, 7] * X[:, 8])
    + 0.9 * np.sin(X[:, 9] ** 3)
    + 0.6 * np.cos(X[:, 10] * X[:, 11] - X[:, 12])
    + 0.4 * np.sin(X[:, 13] + X[:, 14])
    + 0.5 * np.cos(X[:, 15] * X[:, 16])
    + 0.3 * np.sin(X[:, 17] ** 2 - X[:, 18])
    + 0.2 * X[:, 19]
)
y = (y - y.min()) / (y.max() - y.min())
y = 20 + 5 * y
y = y.reshape(-1, 1).astype(np.float32)
# 保存为 tuple 格式
with open("data.txt", "w") as f:
    for i in range(len(X)):
        f.write(f"({', '.join(map(str, X[i]))}, {y[i][0]})\n")
print("✅ 数据已保存到 data.txt，y 由所有 20 个维度共同决定。")
