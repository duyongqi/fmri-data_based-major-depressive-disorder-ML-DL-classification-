import scipy.io as scio
from scipy import stats
import argparse
import os
from multiprocessing import Process, Pool, Manager
import time
import numpy as np
import numba as nb
import pandas as pd
from itertools import product
import sklearn
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import make_classification
import anatomical
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import svm
print(__doc__)
# print("温馨提示：本机为",os.cpu_count(),"核CPU")
# 等待添加新功能：能够实现多进程，加快计算进度,                                  √！！实现了！！
# 注意是将for循环切片，用多核运算 多个for循环，而不是每一个for循环用一个核，这样调度太花时间了
# 绝了我实现的这个没啥意义，直接一行代码就可以搞定，怪我没有认真学习函数的使用
# 并行处理用在了其他地方，在每个人做特征拉伸和连接堆叠的时候，可以并行同时对很多个人做拼接和堆叠
# numba比较适合for循环

# 等待添加新功能，能够直接爬取atlas区域名称，不用一个个复制过来                 √！！实现了！！
# 等待添加新功能，地图集选择，需要做几个地图集对应的DFC                         √！！实现了！！
# 等待解释，做DFC的时候，地图集怎么用在大脑上，按什么顺序                       √！！实现了！！
# 等待解释，为什么一个需要[0, :],一个不需要                                     √！！解决了！！
# 等待解释，再读一遍文章                                                       √！！读了！！
# def t_test_pool(matrix_one, matrix_two, start, length, p_value):
#     # p_value[i] = stats.ttest_ind(matrix_one, matrix_two).pvalue
#     for j in range(length):
#         p_value[start + j] = stats.ttest_ind(a[:, j], b[:, j]).pvalue
@nb.jit()
def dfc2vector(dfc):
    """
    将dfc的每个时间点的一半拉直，返回两种，一维拼接和矩阵拼接，用来做t检验和RNN
    :param dfc: 一个人的动态功能连接文件地址
    :return: t_test_vector：用来做t test 的一维向量的拼接得到的更长的一维向量（每个动态功能连接，也就是每个人得到一个一维向量 ；
    dfc_matrix： 用来做lstm的一维向量沿着时间堆叠成的二位矩阵，也就是每一个人一个矩阵，一个维度代表功能连接点，一个维度代表时间点 ；
    """
    
    # a = scio.loadmat(dfc)
    for i, fc in enumerate(dfc):
        fc_tri = np.triu(fc, 1)
        fc_vector = np.array([])
        for row in fc_tri:
            row_trim_zero = np.trim_zeros(row, 'f')
            fc_vector = np.append(fc_vector, row_trim_zero)
        if i ==0:
            t_test_vector = fc_vector # 拼接成一维向量
            dfc_matrix = fc_vector # 按行拼接成矩阵
        else:
            t_test_vector = np.append(t_test_vector, fc_vector)
            dfc_matrix = np.vstack((dfc_matrix, fc_vector))
    return t_test_vector, dfc_matrix

def t_test_parallel(matrix_one, matrix_two):
    """
    对得到的行代表个体，列代表特征的矩阵，每一列进行双样本t检验，第一个矩阵的一列代表一个类别对应的特征分布，第二个矩阵的对应列代表第二个类别代表的特征分布
    :param matrix_one:多个变量，每个变量的样本是一列
    :param matrix_two:多个变量，每个变量的样本是一列，与matrix_one对应
    :return:
    """
    # 检测两个矩阵列数是否相等，不等则返回提示。等待加入的功能
    # feature_number_2 = matrix_two.shape
    # feature_number_1 = matrix_one.shape
    # if feature_number_1 != feature_number_2:
    #     return '在这里写中断，提醒两个矩阵列数要一致'
    feature_number = matrix_one.shape[1]
    p_value = stats.ttest_ind(matrix_one,matrix_two).pvalue
    # manager = Manager()
    # p_value = manager.Array('f', np.empty(feature_number))
    # p = Pool(processes=20)
    # length = feature_number//50
    # for i in range(50):
    #     start = i*length
    #     end = min(start + length, feature_number)
    #     p.apply_async(t_test_pool, (matrix_one[:, start:end], matrix_two[:, start:end], start, length, p_value))
    #     # p_value[i] = stats.ttest_ind(matrix_one[:, i], matrix_two[:, i]).pvalue
    # p.close()
    # p.join()
    # print('完成tst')
    # print(p_value)
    # p_value1 = len(p_value)
    # print(p_value1)
    return p_value

def find_file(direct, subdirect, filename, suffix):
    '''
    寻找相应目录,目录转换,可以
    '''
    root, name = os.path.split(direct)
    name_direct = ''.join((name, subdirect))
    filename = ''.join((filename, suffix))
    file_direct = os.path.join(root, name_direct, filename)
    return file_direct

def dfc2vector_parallel(direct, fctype):
    """
    实现一个类别的所有人dfc的批量转换并得到整体的矩阵
    :param direct: 存放多个人动态功能连接的文件路径
    :param fctype:功能连接的类型DFC或者SFC
    :return:
    """
    p = Pool(processes=15)
    t_test_vectors = np.array([])
    rnn_matrix = np.array([])
    results = []
    for root, dirs, files in os.walk(direct):
        files.sort()
        for index, name in enumerate(files):
            dfc = os.path.join(root, name)
            temps = []
            if fctype == 'DFC':
                data = scio.loadmat(dfc)['DZStruct'][0, 0]
                for temp in data:
                    temps.append(temp)
                data = np.array(temps)
            elif fctype == 'SFC':
                data = np.loadtxt(dfc)
                data = data[np.newaxis, :]
            # print('正在处理{}组的第{}个人'.format(os.path.split(direct)[1], index))
            result = p.apply_async(dfc2vector,(data,))
            # (t_test_vector, dfc_matrix) = p.apply_async(dfc2vector,(os.path.join(root, name)))
            # (t_test_vector, dfc_matrix) = dfc2vector(os.path.join(root, name), fctype)
            # dfc_matrix = dfc_matrix[np.newaxis, :]
            # if index == 0:
            #     t_test_vectors = t_test_vector
            #     rnn_matrix = dfc_matrix
            # else:
            #     t_test_vectors = np.vstack((t_test_vectors, t_test_vector))
            #     rnn_matrix = np.vstack((rnn_matrix, dfc_matrix))
            results.append(result)
    p.close()
    p.join() 
    for index, result in enumerate(results):
        print('正在处理{}组的第{}个人'.format(os.path.split(direct)[1], index))
        (t_test_vector, dfc_matrix) = result.get()
        dfc_matrix = dfc_matrix[np.newaxis, :]
        if index == 0:
            t_test_vectors = t_test_vector
            rnn_matrix = dfc_matrix
        else:
            t_test_vectors = np.vstack((t_test_vectors, t_test_vector))
            rnn_matrix = np.vstack((rnn_matrix, dfc_matrix))
    # 将得到的群体矩阵写到文件里面
    # 保存 t test的文件
    file_save_SVM = find_file(direct, '_splice_along_time')
    os.mkdir(file_save_SVM)
    file_save_SVM_file = os.path.join(file_save_SVM, 't_test')
    scio.savemat(file_save_SVM_file, {'data': t_test_vectors})
    if fctype == 'DFC':
        # 保存之后做lstm的文件,只有dfc需要保存用来做lstm的文件，Sfc不需要
        file_save_LSTM = find_file(direct, '_stack_along_time')
        os.mkdir(file_save_LSTM)
        file_save_LSTM_file = os.path.join(file_save_LSTM, 'lstm')
        scio.savemat(file_save_LSTM_file, {'data': rnn_matrix})

def matrix2dataframe(filepath, index):
    '''
    把matrix转成dataframe
    '''
    filepath_matrix = scio.loadmat(filepath)['data']
    time = filepath_matrix.shape[1]//len(index)
    time = np.arange(1, time+1)
    time = time.astype(np.str_)
    # a = [i for i in product(time, index)]
    time_anatomical = pd.MultiIndex.from_tuples([i for i in product(time, index)], names=('time_point', 'anatomical'))
    data = pd.DataFrame(data=filepath_matrix, columns=time_anatomical)
    if not os.path.exists(''.join((filepath, '.csv'))):
        data.to_csv(''.join((filepath, '.csv')), chunksize=5000)
    if not os.path.exists(''.join((filepath, '.pkl'))):
        data.to_pickle(''.join((filepath, '.pkl')))
    return data

def data_prepare(dir_path, fctype, anatomical_index, dirtype):
    print('load{}数据'.format(dirtype))
    file_dir = find_file(dir_path, '_splice_along_time', 't_test', '.pkl')
    if os.path.exists(file_dir):
        data = pd.read_pickle(file_dir)
        print('{}加载完毕'.format(dirtype))
    else:
        # 目录转换
        file_dir = find_file(dir_path, '_splice_along_time', 't_test', '')
        # 判断数据文件是否已经存在（是否已经进行过转换）
        if os.path.exists(file_dir):
            pass
        else:
            # 执行转换
            dfc2vector_parallel(dir_path, fctype)
        # load 矩阵
        data = matrix2dataframe(file_dir, anatomical_index)
        print('{}加载完毕'.format(dirtype))
    #index合并
    data.columns = data.columns.map("-".join)
    # 新增一列标签列
    if dirtype == 'HC':
        data = data.assign(label=1)
    else:
        data = data.assign(label=0)
    return data

def SVM_data_prepare(dir_hc, dir_mdd, fctype, anatomical_index):
    """
    直接load拉伸后的dataframe文件，文件不存则进行转换，转换的时候直接load拉伸后的文件用来进行转换，拉伸文件不存在的时候才进行矩阵拉伸
    :param dir_hc: 存放hc动态功能连接或静态功能连接文件的目录
    :param dir_mdd: 存放mdd功能链接文件的目录
    :param fctype:功能连接的类型DFC或者SFC
    :param anatomical_index:拉伸之后的生理标识
    :return:
    """
    try:
        final_data = pd.read_pickle(find_file(os.path.split(dir_hc)[0],
                                             '', 
                                             fctype,
                                             'final_data_svm.pkl'))
        print('load成功')
    except:
        print('直接load失败，开始下一级load')
        data_hc = data_prepare(dir_hc, fctype, anatomical_index, 'HC')
        data_mdd = data_prepare(dir_mdd, fctype, anatomical_index, 'MDD')
        #合并hc和mdd
        final_data = pd.concat([data_hc, data_mdd])
        # print(final_data)
        #打乱
        final_data = sklearn.utils.shuffle(final_data, random_state=1)
        final_data.to_pickle(find_file(os.path.split(dir_hc)[0],
                                             '', 
                                             fctype,
                                             'final_data_svm.pkl'))
    print(final_data)
    #划分特征和label
    train, test = train_test_split(final_data, test_size=0.2, random_state=10)
    return train, test

def load_SVM_data(dir_hc, dir_mdd, fctype, anatomical_index):
    #找到对应的存储训练数据和测试数据的文件夹
    print('load训练数据和测试数据')
    root, _ = os.path.split(dir_hc)
    svm_train_path = os.path.join(root,  fctype, 'svm_train_data')
    svm_test_path = os.path.join(root, fctype, 'svm_test_data',)
    train_path = os.path.join(svm_train_path, 'train.pkl')
    test_path = os.path.join(svm_test_path, 'test.pkl')
    if os.path.exists(train_path) and os.path.exists(test_path):
        train = pd.read_pickle(train_path)
        test = pd.read_pickle(test_path)
    else:
        print('load总数据')
        train, test = SVM_data_prepare(dir_hc, dir_mdd, fctype, anatomical_index)
        #存储在文件夹DFC——train和DFC-test下面
        if not os.path.exists(svm_train_path):
            os.makedirs(svm_train_path)
        if not os.path.exists(svm_test_path):
            os.makedirs(svm_test_path)
        train.to_pickle(train_path)
        test.to_pickle(test_path)
    x_train = train.iloc[:, 0:-1]
    y_train = train.iloc[:, -1]
    x_test = test.iloc[:, 0:-1]
    y_test = test.iloc[:, -1]
    print('训练测试数据加载成功！！=D')
    return x_train, y_train, x_test, y_test
    
def main():
    """
    主函数，三种方法任选一种执行
    :return:
    """
    parser = argparse.ArgumentParser(description='预处理之后数据的抑郁症诊断，可以选择三种方法，一种是DFC+特征选择SVM，一种是DFC+LSTM，一种是直接LSTM')
    parser.add_argument('method', choices=['SVM', 'LSTM', 'oLSTM'],help='分类方法类别')
    parser.add_argument('fctype', choices=['DFC', 'SFC'], help='功能连接类别')
    parser.add_argument('hc', help='正常组FC目录')
    parser.add_argument('mdd', help='MDD组FC目录')
    parser.add_argument('--threshold', '-t', help='t test的阈值', type=float)
    parser.add_argument('--atlas', '-a', default='AAL90', help='选择使用的分割图，注意要和输入的对应的地址中数据使用的地址一致，默认是AAL90')
    args = parser.parse_args()
    # print(args)
    feature_anatomical = anatomical.anatomical_vector_generator(args.atlas)
    # print('fffff')
    if args.method == 'SVM':
        print('###进行SVM分类')
        x_train, y_train, x_test, y_test = load_SVM_data(args.hc, args.mdd, args.fctype, feature_anatomical)

        #像个办法把这调用写道SVM里面，让他可以传参
        root, _ = os.path.split(args.hc)
        model_path = os.path.join(root,  args.fctype, 'model.joblib')
        train_result_path = os.path.join(root,  args.fctype, 'train_result.pkl')

        [clf, clf_cv_result] = svm.train_loadtrain(x_train, y_train, model_path, train_result_path)
        print('绘制特征选择曲线')
        svm.feature_select_draw(clf_cv_result, 'select_fig')
        print('绘制测试数据ROC曲线')
        svm.test_roc(x_test, y_test, 'roc_auc', clf)
        # 再把特征对应到生理特征上（用之前生成的文件）
        if args.fctype == 'DFC':
            timepoint = 125
        else:
            timepoint = 1
        # 时间点的个数，为了获取拉伸拼接后的长度
        index = np.arange(len(feature_anatomical)*timepoint).reshape(1, -1)
        t_test_index = clf.best_estimator_.steps[0][1].transform(index)
        feature_rank = np.array(clf.best_estimator_.steps[1][1].ranking_)
        feature_rank_index = np.where(feature_rank==1)[0]
        feature_index = t_test_index[0][feature_rank_index]
        print('###进行特征的映射，找出特征对应的区域连接')
        feature_time_point = feature_index // len(feature_anatomical)
        feature_index = feature_index % len(feature_anatomical)
        print(feature_index)
        print(feature_time_point)
        anatomical.print_anatomical_unique(feature_index, feature_anatomical)
    elif args.method == 'LSTM':
        print('DFC的时间序列进行LSTM做分类')
        # LSTM
        # 先得到拉伸后沿着时间的矩阵，每个人一个二位矩阵
        # 再做LSTM分类
    else:
        print('脑区分割后的原始时间序列进行LSTM做分类')
        # LSTM_ORIGIN
        # 直接读取nii文件再分割，得到区域对应的序列，最后每个人是一个二位矩阵
        # 再做LSTM分类

if __name__ == "__main__":
    main()
    #调试用
    # a, b, feature_index = t_test("/data/upload/duyongqi/mdd_prediction_brainnetcome/HC_DFC", "/data/upload/duyongqi/mdd_prediction_brainnetcome/MDD_DFC", 0.001, 'DFC')
    # feature_anatomical = anatomical_vector_generator('AAL90')
    # feature_index = feature_index % len(feature_anatomical)
    # anatomical.print_anatomical(feature_index, feature_anatomical)
    # feature_anatomical = anatomical_vector_generator('AAL90')
    # feature_index = np.array(range(0,3))

    # [clf, clf_cv_result] = train_loadtrain(x_train, y_train, 'model.joblib', 'train_result.pkl')
    # list_region = ['1', '2', '3', '4','5', '6']
    # vector = anatomical.get_anatomical(list_region)
    # list_region = ['SFG_L_7_1', 'SFG_R_7_1', 'SFG_L_7_2', 'SFG_R_7_2', 'SFG_L_7_3', 'SFG_R_7_3', 'SFG_L_7_4', 'SFG_R_7_4', 'SFG_L_7_5', 'SFG_R_7_5', 'SFG_L_7_6', 'SFG_R_7_6', 'SFG_L_7_7', 'SFG_R_7_7', 'MFG_L_7_1', 'MFG_R_7_1', 'MFG_L_7_2', 'MFG_R_7_2', 'MFG_L_7_3', 'MFG_R_7_3', 'MFG_L_7_4', 'MFG_R_7_4', 'MFG_L_7_5', 'MFG_R_7_5', 'MFG_L_7_6', 'MFG_R_7_6', 'MFG_L_7_7', 'MFG_R_7_7', 'IFG_L_6_1', 'IFG_R_6_1', 'IFG_L_6_2', 'IFG_R_6_2', 'IFG_L_6_3', 'IFG_R_6_3', 'IFG_L_6_4', 'IFG_R_6_4', 'IFG_L_6_5', 'IFG_R_6_5', 'IFG_L_6_6', 'IFG_R_6_6', 'OrG_L_6_1', 'OrG_R_6_1', 'OrG_L_6_2', 'OrG_R_6_2', 'OrG_L_6_3', 'OrG_R_6_3', 'OrG_L_6_4', 'OrG_R_6_4', 'OrG_L_6_5', 'OrG_R_6_5', 'OrG_L_6_6', 'OrG_R_6_6', 'PrG_L_6_1', 'PrG_R_6_1', 'PrG_L_6_2', 'PrG_R_6_2', 'PrG_L_6_3', 'PrG_R_6_3', 'PrG_L_6_4', 'PrG_R_6_4', 'PrG_L_6_5', 'PrG_R_6_5', 'PrG_L_6_6', 'PrG_R_6_6', 'PCL_L_2_1', 'PCL_R_2_1', 'PCL_L_2_2', 'PCL_R_2_2', 'STG_L_6_1', 'STG_R_6_1', 'STG_L_6_2', 'STG_R_6_2', 'STG_L_6_3', 'STG_R_6_3', 'STG_L_6_4', 'STG_R_6_4', 'STG_L_6_5', 'STG_R_6_5', 'STG_L_6_6', 'STG_R_6_6', 'MTG_L_4_1', 'MTG_R_4_1', 'MTG_L_4_2', 'MTG_R_4_2', 'MTG_L_4_3', 'MTG_R_4_3', 'MTG_L_4_4', 'MTG_R_4_4', 'ITG_L_7_1', 'ITG_R_7_1', 'ITG_L_7_2', 'ITG_R_7_2', 'ITG_L_7_3', 'ITG_R_7_3', 'ITG_L_7_4', 'ITG_R_7_4', 'ITG_L_7_5', 'ITG_R_7_5', 'ITG_L_7_6', 'ITG_R_7_6', 'ITG_L_7_7', 'ITG_R_7_7', 'FuG_L_3_1', 'FuG_R_3_1', 'FuG_L_3_2', 'FuG_R_3_2', 'FuG_L_3_3', 'FuG_R_3_3', 'PhG_L_6_1', 'PhG_R_6_1', 'PhG_L_6_2', 'PhG_R_6_2', 'PhG_L_6_3', 'PhG_R_6_3', 'PhG_L_6_4', 'PhG_R_6_4', 'PhG_L_6_5', 'PhG_R_6_5', 'PhG_L_6_6', 'PhG_R_6_6', 'pSTS_L_2_1', 'pSTS_R_2_1', 'pSTS_L_2_2', 'pSTS_R_2_2', 'SPL_L_5_1', 'SPL_R_5_1', 'SPL_L_5_2', 'SPL_R_5_2', 'SPL_L_5_3', 'SPL_R_5_3', 'SPL_L_5_4', 'SPL_R_5_4', 'SPL_L_5_5', 'SPL_R_5_5', 'IPL_L_6_1', 'IPL_R_6_1', 'IPL_L_6_2', 'IPL_R_6_2', 'IPL_L_6_3', 'IPL_R_6_3', 'IPL_L_6_4', 'IPL_R_6_4', 'IPL_L_6_5', 'IPL_R_6_5', 'IPL_L_6_6', 'IPL_R_6_6', 'PCun_L_4_1', 'PCun_R_4_1', 'PCun_L_4_2', 'PCun_R_4_2', 'PCun_L_4_3', 'PCun_R_4_3', 'PCun_L_4_4', 'PCun_R_4_4', 'PoG_L_4_1', 'PoG_R_4_1', 'PoG_L_4_2', 'PoG_R_4_2', 'PoG_L_4_3', 'PoG_R_4_3', 'PoG_L_4_4', 'PoG_R_4_4', 'INS_L_6_1', 'INS_R_6_1', 'INS_L_6_2', 'INS_R_6_2', 'INS_L_6_3', 'INS_R_6_3', 'INS_L_6_4', 'INS_R_6_4', 'INS_L_6_5', 'INS_R_6_5', 'INS_L_6_6', 'INS_R_6_6', 'CG_L_7_1', 'CG_R_7_1', 'CG_L_7_2', 'CG_R_7_2', 'CG_L_7_3', 'CG_R_7_3', 'CG_L_7_4', 'CG_R_7_4', 'CG_L_7_5', 'CG_R_7_5', 'CG_L_7_6', 'CG_R_7_6', 'CG_L_7_7', 'CG_R_7_7', 'MVOcC _L_5_1', 'MVOcC _R_5_1', 'MVOcC _L_5_2', 'MVOcC _R_5_2', 'MVOcC _L_5_3', 'MVOcC _R_5_3', 'MVOcC _L_5_4', 'MVOcC _R_5_4', 'MVOcC _L_5_5', 'MVOcC _R_5_5', 'LOcC_L_4_1', 'LOcC_R_4_1', 'LOcC _L_4_2', 'LOcC _R_4_2', 'LOcC _L_4_3', 'LOcC _R_4_3', 'LOcC_L_4_4', 'LOcC_R_4_4', 'LOcC_L_2_1', 'LOcC_R_2_1', 'LOcC_L_2_2', 'LOcC_R_2_2', 'Amyg_L_2_1', 'Amyg_R_2_1', 'Amyg_L_2_2', 'Amyg_R_2_2', 'Hipp_L_2_1', 'Hipp_R_2_1', 'Hipp_L_2_2', 'Hipp_R_2_2', 'BG_L_6_1', 'BG_R_6_1', 'BG_L_6_2', 'BG_R_6_2', 'BG_L_6_3', 'BG_R_6_3', 'BG_L_6_4', 'BG_R_6_4', 'BG_L_6_5', 'BG_R_6_5', 'BG_L_6_6', 'BG_R_6_6', 'Tha_L_8_1', 'Tha_R_8_1', 'Tha_L_8_2', 'Tha_R_8_2', 'Tha_L_8_3', 'Tha_R_8_3', 'Tha_L_8_4', 'Tha_R_8_4', 'Tha_L_8_5', 'Tha_R_8_5', 'Tha_L_8_6', 'Tha_R_8_6', 'Tha_L_8_7', 'Tha_R_8_7', 'Tha_L_8_8', 'Tha_R_8_8']
    # vector = anatomical.anatomical_vector_generator('AAL90')
    # print('###SVM+DFC抑郁症诊断，挑选最佳分类特征子集')
    # x_train, y_train, x_test, y_test = load_SVM_data("/data/upload/duyongqi/mdd_prediction_aal/mdd/HC_DFC",
    #                                                 "/data/upload/duyongqi/mdd_prediction_aal/mdd/MDD_DFC", 
    #                                                 'DFC', vector)
    # print('###进行SVM分类')
    # #像个办法把这调用写道SVM里面，让他可以传参
    # root, _ = os.path.split("/data/upload/duyongqi/mdd_prediction_brainnetcome/HC_DFC/")
    # model_path = os.path.join(root,  'DFC', 'model.joblib')
    # train_result_path = os.path.join(root,  'DFC', 'train_result.pkl')

    # [clf, clf_cv_result] = svm.train_loadtrain(x_train, y_train, model_path, train_result_path)
    # # svm.feature_select_draw(clf_cv_result, 'select_fig')
    # # svm.test_roc(x_test, y_test, 'roc_auc', clf)
    # # 再把特征对应到生理特征上（用之前生成的文件）
    # a = 'DFC'
    # if a == 'DFC':
    #     timepoint = 125
    # else:
    #     timepoint = 1
    # # 释奠奠的个数，为了获取拉伸拼接后的长度
    # index = np.arange(len(vector)*timepoint).reshape(1, -1)
    # t_test_index = clf.best_estimator_.steps[0][1].transform(index)
    # feature_rank = np.array(clf.best_estimator_.steps[1][1].ranking_)
    # feature_rank_index = np.where(feature_rank==1)[0]
    # feature_index = t_test_index[0][feature_rank_index]
    # print('###进行特征的映射，找出特征对应的区域连接')
    # feature_time_point = feature_index // len(vector)
    # feature_index = feature_index % len(vector)
    # print(feature_index)
    # print(feature_time_point)
    # anatomical.print_anatomical_unique(feature_index, vector)