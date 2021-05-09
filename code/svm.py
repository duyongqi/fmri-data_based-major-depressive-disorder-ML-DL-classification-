from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold, train_test_split
from sklearn.feature_selection import RFE, SelectPercentile, f_classif
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import os
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from joblib import dump, load
from draw import feature_select_draw, test_roc
#把选出来的特征编号找出来
def train_loadtrain(train_data, train_label, filepath_save, result_path):
    if os.path.exists(filepath_save):
        clf = load(filepath_save)
        cv_results = pd.read_pickle(result_path) 
    else:
        selector = SelectPercentile(percentile=1)
        classifier = SVC(kernel='linear')
        rfe_clf = RFE(classifier, step=50, verbose=1)
        steps = [('reduction',selector), ("selector", rfe_clf), ("SVC", SVC(kernel='rbf'))]
        pipe = Pipeline(steps = steps)
        tuned_parameters = {'selector__n_features_to_select':[15, 30, 50, 100, 200],
                            'SVC__C':[0.1, 1, 10, 100],
                            'SVC__gamma':[0.0001, 0.001, 0.01, 0.1]               
                            }
        clf = GridSearchCV(pipe, tuned_parameters, scoring='roc_auc',
                            cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=1,random_state=36851234),
                            verbose=1, refit=True, n_jobs=10)
        clf.fit(train_data, train_label)
        cv_results = pd.DataFrame.from_dict(clf.cv_results_)
        dump(clf, filepath_save) 
        pd.DataFrame.from_dict(clf.cv_results_).to_pickle(result_path)
    return clf, cv_results

if __name__ == "__main__":
    X, y = make_classification(n_samples = 100, n_features = 10, n_informative = 8,
                                n_redundant = 1, n_repeated = 1,
                                n_clusters_per_class = 2, class_sep = 0.5,
                                random_state = 1000, shuffle = False)
    labels = [f"Feature {ii+1}" for ii in range(X.shape[1])]
    X = pd.DataFrame(X, columns = labels)
    y = pd.DataFrame(y, columns = ["Target"])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    [clf, clf_cv_result] = train_loadtrain(x_train, y_train, 'model.joblib', 'train_result.pkl')
    # feature_select_draw(clf_cv_result, 'select_fig')
    # test_roc(x_test, y_test, 'roc_auc', clf)
    print(clf.best_estimator_.steps[0][1].support_)
    
# print(clf.best_params_)
# pipe.set_params(clf.best_params_)

# estimator_rfe = RFECV(pipe, cv = 5, step = 1, scoring = "roc_auc", verbose = 1 )
# gridsearch
# tuned_parameters = {'SVC__C':[0.1, 1, 10, 100, 1000],
#                     'SVC__gamma':[0.0001, 0.001, 0.01, 0.1, 1],                        
#                     'rfe__n_features_to_select':[50, 100, 150, 200, 250, 300]
#                     }
# clf = GridSearchCV(
#     pipe, tuned_parameters, scoring='roc_auc', refit=True)
# clf.fit(iris.data, iris.target)
# a = pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score')
# print(a)

# iris = datasets.load_breast_cancer()




'''
文章的意思：先划分成训练集和测试集，
对训练集的数据做T TEST,留下了5000个特征
对T TEST后的训练集的数据根据SVM-RFE选择最佳特征个数和对应的特征编号，（RFECV？自动选择了最佳的特征个数，但是这个SVM-RFE的SVM的参数是怎么选的，直接用的默认的？）
在选择出来的特征子集上又用gridsearchcv找最佳的SVM分类参数

我看的另一个：
先用gridsearchccv在训练集上找最佳的SVM参数，
基于这个参数构造SVM-RFE，用RFECV找最佳特征子集和最佳特征个数，最后用这个SVM和最佳特征个数的RFE在整个训练集上训练一遍，用在测试集上得到最终的评估分数
'''
