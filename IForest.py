import numpy as np
import pandas as pd
from itertools import starmap
from sklearn.metrics import confusion_matrix

class IsolationForest:
    """
    孤立森林算法
    -------------------------
    subsample_size:子样本集的大小
    n_tree:孤立森林中iTree的数量
    trees:iTree的集合
    c:异常分数中路径长度关于subsample_size的平均值
    -------------------------
    fit():输入二维观察值矩阵X。重复n_trees次，从X中不重复抽样subsample_size个
          样本组成子集，并通过IsolationTree训练iTree，最终得到iTree集合trees

    tree_avg_length():用于计算路径长度的调整项，和异常分数的路径长度平均值

    path_length():将_path_len函数运用到数据集X的每个样本上

    _path_len():通过tree_path_length函数，计算一个样本点在所有iTree的路径长度，求均值

    tree_path_length():通过递归结构，凭借inNode节点上的信息，不断访问新节点，
                       每到一个新节点path_length+1，并在exNode计算最终值。

    anomaly_score():通过每个样本的路径长度计算相应的异常分数

    judge_from_anomaly_score():通过异常分数与阈值判断样本点是否异常

    predict():输入数据集X和阈值，预测数据集X的异常点。
    """
    def __init__(self,subsample_size,n_trees=100):
        self.subsample_size = subsample_size
        self.n_trees = n_trees
        self.trees = []
        self.c = self.tree_avg_length(self.subsample_size)

    def fit(self,X:np.ndarray):
        if isinstance(X,pd.DataFrame):
            X = X.values

        height_limit = int(np.ceil(np.log2(self.subsample_size)))

        for num in range(self.n_trees):
            # subsample from X
            subsample = np.random.choice(X.shape[0],
                                         size=self.subsample_size,
                                         replace=False)
            tree = IsolationTree(height_limit)
            tree.fit(X[subsample])
            self.trees.append(tree)

        return self

    def tree_avg_length(self,size):
        if size > 2:
            harmonic_number = np.log(size - 1.) + 0.5772156649
            return (2. * harmonic_number) - (2. * (size - 1.) /size )
        elif size == 2:
            return 1.
        else:
            return 0.

    def path_length(self,X:np.ndarray) -> np.ndarray:
        if isinstance(X,pd.DataFrame):
            X = X.values

        paths = np.zeros(X.shape[0],dtype=float)
        paths = np.apply_along_axis(self._path_len, 1, X)   #将_path_len运用到
        return np.array(paths)

    def _path_len(self,x):
        length = 0.0
        for tree in self.trees:
            length += self.tree_path_length(x,tree.root,0)
        mean = length / self.n_trees
        return mean

    def tree_path_length(self,x,node,current_PathLength):
        if isinstance(node,exNode):
            return current_PathLength + self.tree_avg_length(node.size)

        split_at = node.split_feature

        if x[split_at] < node.split_value:
            return self.tree_path_length(x,node.left,current_PathLength+1)
        else:
            return self.tree_path_length(x,node.right,current_PathLength+1)

    def anomaly_score(self,X:np.ndarray) -> np.ndarray:
        path_lengths = self.path_length(X)
        return 2.0 ** (-path_lengths / self.c)

    def judge_from_anomaly_score(self,scores,threshold):
        s = scores.copy()
        s[s < threshold] = 0
        s[s >= threshold] = 1
        return s

    def predict(self,X:np.ndarray,threshold:float) -> np.ndarray:
        return self.judge_from_anomaly_score(self.anomaly_score(X), threshold)

class IsolationTree:
    """
    孤立树（iTree）算法
    -----------------------
    height_limit:控制树的高度，hlim越小，iTree越简单，异常分数颗粒度越小
    -----------------------
    fit():输入二维观察值矩阵，创建一棵iTree

    make_nodes():通过递归结构，不断创建新的节点，并在节点中储存分裂后两个节点的信息，最终形成iTree。
    其中，由于在计算路径长度时，会将超过height_limit的部分通过公式计算出平均值，
    所以在该算法中为节省计算资源，会将超过height_limit还未分类的样本点都归于一个叶子节点中。
    """
    def __init__(self,height_limit):
        self.height_limit = height_limit
        self.n_nodes = 0

    def fit(self,X:np.ndarray):
        self.root = self.make_nodes(X,0)

    def make_nodes(self,X,current_PathLength):
        self.n_nodes += 1
        if current_PathLength >= self.height_limit or X.shape[0] <= 1:
            return exNode(X.shape[0])
        else:
            q = np.random.randint(0,X.shape[1])
            minimum = X[:,q].min()
            maximum = X[:,q].max()
            p = np.random.uniform(minimum,maximum)
            leftX = X[X[:,q]<p]
            rightX = X[X[:,q]>=p]

        left = self.make_nodes(leftX,current_PathLength+1)
        right = self.make_nodes(rightX,current_PathLength+1)
        return inNode(left,right,q,p)

class inNode:
    """
    iTree的内部节点，依据分割特征的分割值将数据分为左右两部分
    ----------------------
    left:该节点的左分支树
    right:该节点的右分支树
    split_feature:该节点的分割特征
    split_value:该节点的分割值
    """
    def __init__(self,left,right,split_feature,split_value):
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value

class exNode:
    """
    iTree的叶子节点
    ----------------------
    size:叶子节点中样本点数量
    """
    def __init__(self,size):
        self.size = size


