import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score



def auc_calculation(label,score):
    """
    AUC计算算法，通过公式直接计算auc
    跟sklearn.metrics.roc_auc_score效果一样
    ---------------------
    score:分类算法对样本点的预测值或异常分数
    label:数据集中样本的分类，0表示正常点，1表示异常点
    """
    data = {'score': list(score), 'label': list(label)}
    zip = pd.DataFrame(data).sort_values(by='score', ascending=True).reset_index(drop=True)
    zip['rank'] = zip['score'].rank(axis=0, method='average')

    pos_num = np.sum(zip['label'] == 1)
    neg_num = np.sum(zip['label'] == 0)

    auc = (np.sum(zip['rank'][zip['label'] == 1])-pos_num*(pos_num+1)/2)/(pos_num*neg_num)
    return auc

def indiccators(score,label,threshold):
    pre_bool = score > threshold
    pre = pre_bool.astype(np.int32)

    auc = auc_calculation(label,score) #跟sklearn.metrics.roc_auc_score效果一样
    AP = average_precision_score(label,score)
    recall = recall_score(label,pre)
    precision = precision_score(label,pre)
    f1 = f1_score(label,pre)
    # accuracy = accuracy_score(label,pre)

    return auc,AP,recall,precision,f1


def anomaly_count(score,threshold):
    boolarr = score>threshold
    return boolarr.sum()/len(score)


