import os
import pandas as pd
import numpy as np
root=os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path=root+"/data/"
save_path = root+'/results/'

# 思路1：TF-IDF + 机器学习分类器
# 直接使用TF-IDF对文本提取特征，并使用分类器进行分类。在分类器的选择上，可以使用SVM、LR、或者XGBoost。
#
# 思路2：FastText
# FastText是入门款的词向量，利用Facebook提供的FastText工具，可以快速构建出分类器。
#
# 思路3：WordVec + 深度学习分类器
# WordVec是进阶款的词向量，并通过构建深度学习分类完成分类。深度学习分类的网络结构可以选择TextCNN、TextRNN或者BiLSTM。
#
# 思路4：Bert词向量
# Bert是高配款的词向量，具有强大的建模学习能力。

#评价标准为类别f1_score的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。
#
# 数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4,
#                 '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11,
#                 '彩票': 12, '星座': 13}