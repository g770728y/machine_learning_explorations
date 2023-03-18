from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据集
# 共150个样本, 每个样本包含4特征1标签
# 花萼长度（sepal length）：以厘米为单位，测量鸢尾花萼片的长度。
# 花萼宽度（sepal width）：以厘米为单位，测量鸢尾花萼片的宽度。
# 花瓣长度（petal length）：以厘米为单位，测量鸢尾花花瓣的长度。
# 花瓣宽度（petal width）：以厘米为单位，测量鸢尾花花瓣的宽度。
# 鸢尾花的类别（class ）：鸢尾花可以分为三个类别：Setosa、Versicolor和Virginica。
iris = load_iris()
# 特征矩阵大写
X = iris.data    # 四特征
# 目标变量小写
y = iris.target  # 对应的1标签

# 划分训练集与测试集
# X_train & y_train 表示 train集的特征与标签
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("训练集长度:", len(X_train))
# 定义KNN算法类


class KNN(object):
  def __init__(self, k):
    self.k = k

  # KNN是直接比较测试集与训练集, 并不需要训练过程
  # 输入 训练集的 X/y, 和测试集的X，应当输出 与 y_test 接近的y_pred
  def predict(self, X_train, y_train, X_test):
    y_pred = []

    # 对测试集中的每个样本特征(4个数)
    for x in X_test:
      # 计算待预测样本与所有训练样本之间的距离
      # 最好把这句话在numpy中尝试, 理解为什么axis=1(沿着列方向求和)
      distances = np.sqrt(np.sum(np.square(X_train - x), axis=1))
      # 选取距离最近的k个训练样本, 注意实际需要的是训练样本的index, 所以用argsort
      nearest = np.argsort(distances)[:self.k]
      # 统计k个训练样本中出现最多的标签
      # bindcount很有意思，实际上返回了一个稀疏数组, 得到counts后, 可根据 counts[n] 查 n 出现的次数
      # 注意y_train[nearest]的用法, 仅当二者为np.array时才能用, 否则需要改用take函数
      counts = np.bincount(y_train[nearest])
      # 预测待预测样本的标签
      y_pred.append(np.argmax(counts))

    return y_pred


knn = KNN(k=3)
y_pred = knn.predict(X_train, y_train, X_test)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(y_test)
print(y_pred)
print(accuracy)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
