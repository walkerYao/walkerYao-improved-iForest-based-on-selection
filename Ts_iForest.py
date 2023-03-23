import random
import datetime
import numpy as np
import pandas as pd
from Evaluation import auc_calculation
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from itertools import starmap

class IsolationForest:
    """
    孤立森林算法
    -------------------------
    subsample_size: 子样本集的大小
    n_tree:         孤立森林中iTree的数量
    trees:          iTree的集合
    c:              异常分数中路径长度关于subsample_size的平均值
    -------------------------
    fit():
        输入二维观察值矩阵X。重复n_trees次，从X中不重复抽样subsample_size个样本组成子集，并通过IsolationTree训练iTree，最终得到iTree集合trees

    tree_avg_length():
        用于计算路径长度的调整项，和异常分数的路径长度平均值

    path_length():
        将_path_len函数运用到数据集X的每个样本上

    _path_len():
        通过tree_path_length函数，计算一个样本点在所有iTree的路径长度，求均值

    tree_path_length():
        通过递归结构，凭借inNode节点上的信息，不断访问新节点，每到一个新节点path_length+1，并在exNode计算最终值。

    anomaly_score():
        通过每个样本的路径长度计算相应的异常分数

    judge_from_anomaly_score():
        通过异常分数与阈值判断样本点是否异常

    predict():
        输入数据集X和阈值，预测数据集X的异常点。
    """
    def __init__(self, subsample_size, n_trees=100):
        self.subsample_size = subsample_size
        self.n_trees = n_trees
        self.trees = []
        self.c = self.tree_avg_length(self.subsample_size)

    def fit(self, X: np.ndarray):
        if isinstance(X, pd.DataFrame):
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

    def tree_avg_length(self, size):
        if size > 2:
            harmonic_number = np.log(size - 1.) + 0.5772156649
            return (2. * harmonic_number) - (2. * (size - 1.) / size)
        elif size == 2:
            return 1.
        else:
            return 0.

    def path_length(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        paths = np.zeros(X.shape[0], dtype=float)
        paths = np.apply_along_axis(self._path_len, 1, X)  # 将_path_len运用到
        return np.array(paths)

    def _path_len(self, x):
        length = 0.0
        for tree in self.trees:
            length += self.tree_path_length(x, tree.root, 0)
        mean = length / self.n_trees
        return mean

    def tree_path_length(self, x, node, current_PathLength):
        if isinstance(node, exNode):
            return current_PathLength + self.tree_avg_length(node.size)

        split_at = node.split_feature

        if x[split_at] < node.split_value:
            return self.tree_path_length(x, node.left, current_PathLength + 1)
        else:
            return self.tree_path_length(x, node.right, current_PathLength + 1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        path_lengths = self.path_length(X)
        return 2.0 ** (-path_lengths / self.c)

    def judge_from_anomaly_score(self, scores, threshold):
        s = scores.copy()
        s[s < threshold] = 0
        s[s >= threshold] = 1
        return s

    """精确度度量:auc"""
    def accuracy(self,score, label):
        data = {'score': list(score), 'label': list(label)}
        zip = pd.DataFrame(data).sort_values(by='score', ascending=True).reset_index(drop=True)
        zip['rank'] = zip['score'].rank(axis=0, method='average')

        pos_num = np.sum(zip['label'] == 1)
        neg_num = np.sum(zip['label'] == 0)

        auc = (np.sum(zip['rank'][zip['label'] == 1]) - pos_num * (pos_num + 1) / 2) / (pos_num * neg_num)
        return auc

    """差异性度量"""
    def difference(self,pre1, pre2):
        [[a,c],[b,d]] = confusion_matrix(pre1, pre2)

        p1 = (a + d) / (a + b + c + d)
        p2 = ((a + b) * (a + c) + (b + d) * (c + d)) / (a + b + c + d)**2
        K = 1 - (p1 - p2) / (1 - p2)  # 差异性度量指标一
        return K

    def iTree_score(self,X:np.ndarray,tree):
        path_lengths = starmap(self.tree_path_length,[(X[i, :], tree.root, 0) for i in np.arange(len(X))])
        scores = 2.0 ** (-np.array(list(path_lengths) / self.c))
        return scores

    def iTree_predict(self,X:np.ndarray,tree,threshold):
        path_lengths = starmap(self.tree_path_length,[(X[i, :], tree.root, 0) for i in np.arange(len(X))])
        scores = 2.0 ** (-np.array(list(path_lengths) / self.c))
        return self.judge_from_anomaly_score(scores, threshold)

    def tabu_search(self, data, label, threshold=0.70, iter_threshold=40, timelimit=20, field_num=20, fitness_dif_rate=0.5, accuracy_method='auc'):
        tabu_list = []  # 禁忌列表
        score_result = []
        pre_result = []
        index = [i for i in range(len(self.trees))]
        for tree in self.trees:
            score_result.append(self.iTree_score(data, tree))
            pre_result.append(self.iTree_predict(data, tree, threshold))

        curbest_index = random.choice(index)
        best_pre = pre_result[curbest_index]
        best_scores = score_result[curbest_index]
        start = datetime.datetime.now()

        k = 0
        while k < iter_threshold:
            """元素的禁忌期限一到，从禁忌表中释放该元素"""
            if k >= timelimit:
                tabu_list.pop(0)
            """计算不在禁忌表的任意20棵iTree的fitness值"""
            curindex_list = [i for i in index if i not in tabu_list]
            curindex_list.remove(curbest_index)
            curindex_list = random.sample(curindex_list,field_num)
            fitness_list = []

            for tree_index in curindex_list:
                pre = pre_result[tree_index]
                scores = score_result[tree_index]
                if accuracy_method == 'auc':
                    fitness = fitness_dif_rate * self.difference(best_pre, pre) + (1-fitness_dif_rate) * auc_calculation(label, scores)
                elif accuracy_method == 'ap':
                    fitness = fitness_dif_rate * self.difference(best_pre, pre) + (1-fitness_dif_rate) * average_precision_score(label, scores)
                fitness_list.append(fitness)

            """计算适应度值最高的iTree的预测值，用于计算当前最佳iTree的fitness值"""
            pre = pre_result[curindex_list[fitness_list.index(max(fitness_list))]]
            if accuracy_method == 'auc':
                best_fitness = fitness_dif_rate * self.difference(best_pre, pre) + (1-fitness_dif_rate) * auc_calculation(label, best_scores)
            elif accuracy_method == 'ap':
                best_fitness = fitness_dif_rate * self.difference(best_pre, pre) + (1-fitness_dif_rate) * average_precision_score(label, best_scores)

            if max(fitness_list) > best_fitness:
                tabu_list.append(curindex_list[fitness_list.index(max(fitness_list))])
                k += 1
            else:
                tabu_list.append(curbest_index)
                curbest_index = curindex_list[fitness_list.index(max(fitness_list))]
                best_pre = pre
                best_scores = score_result[curbest_index]
                k += 1
            print('current_best_tree:', curbest_index, 'current_best_fitness:', best_fitness)
            print('Progress:%d/%d, totally time is' % (k,iter_threshold), datetime.datetime.now() - start)

        tabu_list.append(curbest_index)
        selected_trees = [self.trees[i] for i in tabu_list]

        self.trees = selected_trees
        # score = self.anomaly_score(data)
        # print('tabu_forest的AUC:', auc_calculation(score, label))

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
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
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.n_nodes = 0

    def fit(self, X: np.ndarray):
        self.root = self.make_nodes(X, 0)

    def make_nodes(self, X, current_PathLength):
        self.n_nodes += 1
        if current_PathLength >= self.height_limit or X.shape[0] <= 1:
            return exNode(X.shape[0])
        else:
            q = np.random.randint(0, X.shape[1])
            minimum = X[:, q].min()
            maximum = X[:, q].max()
            p = np.random.uniform(minimum, maximum)
            leftX = X[X[:, q] < p]
            rightX = X[X[:, q] >= p]

        left = self.make_nodes(leftX, current_PathLength + 1)
        right = self.make_nodes(rightX, current_PathLength + 1)
        return inNode(left, right, q, p)


class inNode:
    """
    iTree的内部节点，依据分割特征的分割值将数据分为左右两部分
    ----------------------
    left:该节点的左分支树
    right:该节点的右分支树
    split_feature:该节点的分割特征
    split_value:该节点的分割值
    """
    def __init__(self, left, right, split_feature, split_value):
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
    def __init__(self, size):
        self.size = size