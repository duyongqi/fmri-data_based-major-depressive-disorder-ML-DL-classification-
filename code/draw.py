import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold, train_test_split
from sklearn.feature_selection import RFE, SelectPercentile, f_classif
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
def corr_matrix_generate(size):
    '''随机生成一个相关矩阵'''
    X_train = shuffle(pd.DataFrame(np.arange(size)), random_state=0)
    # X_train1 = shuffle(pd.DataFrame(np.arange(100)), random_state=2)
    # X_train2 = shuffle(pd.DataFrame(np.arange(100)), random_state=3)
    # X_train3 = shuffle(pd.DataFrame(np.arange(100)), random_state=10)
    # X_train4 = shuffle(pd.DataFrame(np.arange(100)), random_state=20)
    # , X_train1, X_train2, X_train3, X_train4
    # X_train = pd.concat([X_train0])
    a = np.array(X_train.values).reshape(size//10,10)
    b = pd.DataFrame(a)
    corr_matrix = b.corr(method = "spearman").abs()
    for i in range(10):
        corr_matrix.iloc[i,i] = 0
        for j in range(i):
            corr_matrix.iloc[i,j] = 0
    sns.set(font_scale = 1.0)
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix.iloc[:,1], cmap= "YlGnBu", ax = ax)
    f.tight_layout()
    plt.show()
    plt.savefig("correlation_matrix.png", dpi = 1080)
# def draw():
#     result = pd.read_pickle("/home/duyongqi/mddprediction_fmri/testresult.pkl")
#     result_draw = result.loc[:,('param_SVC__C', 'param_SVC__gamma', 'param_selector__n_features_to_select', 'mean_test_score', 'std_test_score')]
#     print(result_draw.head(10))
#     g = sns.FacetGrid(result_draw, row="param_SVC__C", col='param_SVC__gamma', margin_titles=True)
#     g.map(sns.lineplot, 'param_selector__n_features_to_select', 'mean_test_score', color='m')
#     g.set_axis_labels("feature_select_number", "ROC")
#     plt.show()
#     g.savefig('parame_select')

def feature_select_draw(result, figpath_select):
    # 绘制特征选择和参数选择过程
    # result = pd.read_pickle(result_path)
    plt.figure()
    result_draw = result.loc[:,('param_SVC__C', 'param_SVC__gamma', 'param_selector__n_features_to_select', 'mean_test_score', 'std_test_score')]
    # print(result_draw.head(10))
    sns.set_theme(style="darkgrid")
    g = sns.FacetGrid(result_draw, row="param_SVC__C", col='param_SVC__gamma', margin_titles=True)
    g.map(sns.lineplot, 'param_selector__n_features_to_select', 'mean_test_score', color='m')
    g.set_axis_labels("feature_select_number", "ROC")
    plt.show()
    g.savefig(figpath_select)

def test_roc(test_data, test_label, figpath_roc, clf):
    # 绘制ROC曲线，用最优参数
    plt.figure()
    y_result = clf.best_estimator_.decision_function(test_data)
    fpr, recall, thresholds = roc_curve(y_true=test_label,  # 真实标签是
                                y_score=y_result,  # 置信度，也可以是概率值
                                pos_label=1,
                                drop_intermediate=False)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x=fpr, y=recall, palette='Set2', label='ROC curve', estimator=None)
    plt.show()
    plt.savefig(figpath_roc)


if __name__ == "__main__":
    draw()