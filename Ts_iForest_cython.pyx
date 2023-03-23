# cython: language_level=3

import random
import datetime
import numpy as np
import pandas as pd
from Evaluation import auc_calculation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from Evaluation import indiccators
from itertools import starmap
cimport numpy as np
cimport cython

cdef class IsolationForest:
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

    difference():
        差异性度量

    iTree_score():
        单棵iTree的异常分数

    iTree_predict():
        单棵iTree的预测结果

    tabu_search():
        禁忌搜索，得到差异性大且精确度高的iTree集合，并更新隔离森林

    predict():
        输入数据集X和阈值，预测数据集X的异常点。
    """

    cdef int subsample_size
    cdef int n_trees
    cdef list trees
    cdef float c

    def __init__(self,int subsample_size,int n_trees=100):
        self.subsample_size = subsample_size
        self.n_trees = n_trees
        self.trees = []
        self.c = self.tree_avg_length(subsample_size)

    def fit(self,np.ndarray X):

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

    cdef tree_avg_length(self,int size):
        cdef double harmonic_number
        if size > 2:
            harmonic_number = np.log(size - 1.) + 0.5772156649
            return (2. * harmonic_number) - (2. * (size - 1.) / size)
        elif size == 2:
            return 1.
        else:
            return 0.

    def path_length(self,np.ndarray X):
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

    def anomaly_score(self,np.ndarray X):
        path_lengths = self.path_length(X)
        return 2.0 ** (-path_lengths / self.c)

    def judge_from_anomaly_score(self, scores, threshold):
        s = scores.copy()
        s[s < threshold] = 0
        s[s >= threshold] = 1
        return s

    """精确度度量:auc"""
    # 使用auc_calculation

    """差异性度量"""
    cdef float difference(self,np.ndarray pre1,np.ndarray pre2):
        cdef int a, b, c, d
        cdef float p1, p2, K

        [[a,c],[b,d]] = confusion_matrix(pre1, pre2)

        p1 = (a + d) / (a + b + c + d)
        p2 = ((a + b) * (a + c) + (b + d) * (c + d)) / (a + b + c + d)**2
        K = 1 - (p1 - p2) / (1 - p2)  # 差异性度量指标一
        return K

    def iTree_score(self,np.ndarray X,object tree):
        cdef np.ndarray scores

        path_lengths = starmap(self.tree_path_length,[(X[i, :], tree.root, 0) for i in np.arange(len(X))])
        scores = 2.0 ** (-np.array(list(path_lengths)) / self.c)
        return scores

    def  iTree_predict(self,X:np.ndarray,tree,threshold):
        cdef np.ndarray scores

        path_lengths = starmap(self.tree_path_length,[(X[i, :], tree.root, 0) for i in np.arange(len(X))])
        scores = 2.0 ** (-np.array(list(path_lengths)) / self.c)
        return self.judge_from_anomaly_score(scores, threshold)

    def tabu_search(self, data, label, threshold=0.55, iter_threshold=40, timelimit=20, field_num=20, fitness_dif_rate=0.5, accuracy_method='auc'):
        """
            data:数据集
            label:标签值（是否异常），1为异常，0为正常
            threshold:阈值，结合异常分数判断是否异常值，
            iter_threshold:迭代次数
            timelimit:禁忌列表的长度，亦为最终保留的iTree棵树-1
            field_num:领域大小
            fitness_dif_rate:适应度函数中差异性度量的权重
            accuracy_method:适应度函数中精确度度量方法
        """
        tabu_list = []  # 禁忌列表
        score_result = []
        pre_result = []
        index = [i for i in range(len(self.trees))]
        for tree in self.trees:
            score_result.append(self.iTree_score(data, tree))
            pre_result.append(self.iTree_predict(data, tree, threshold))

        curbest_index = random.choice(index)
        cdef np.ndarray best_pre = pre_result[curbest_index]
        cdef np.ndarray best_scores = score_result[curbest_index]
        cdef np.ndarray pre
        cdef np.ndarray scores
        start = datetime.datetime.now()

        print('已更新62')
        cdef int k = 0
        while k < iter_threshold:
            """元素的禁忌期限一到，从禁忌表中释放该元素"""
            if k >= timelimit:
                tabu_list.pop(0)
            """计算不在禁忌表的所有iTree的fitness值"""
            curindex_list = [i for i in index if i not in tabu_list]
            curindex_list.remove(curbest_index)
            curindex_list = random.sample(curindex_list, field_num)
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
            print('current_best_tree:',curbest_index,'current_best_fitness:',best_fitness)
            print('Total_Progress:%d/%d, totally time is' % (k,iter_threshold), datetime.datetime.now() - start)

        tabu_list.append(curbest_index)
        selected_trees = [self.trees[i] for i in tabu_list]
        self.trees = selected_trees
        # score = self.anomaly_score(data)
        # print('tabu_forest的AUC:', auc_calculation(score, label))
        # print('tabu_forest的指标系统:')
        # indiccators(score, label, 0.90)

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

    def make_nodes(self,np.ndarray X,int current_PathLength):
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