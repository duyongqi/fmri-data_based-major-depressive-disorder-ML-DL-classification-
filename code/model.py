'''
模型编写
'''

# def t_test(dir_hc, dir_mdd, threshold, fctype):
#     """
#     进行t test，直接load文件，文件不存在的时候才进行矩阵拉伸
#     :param dir_hc: 存放hc动态功能连接文件的目录
#     :param dir_mdd: 存放mdd功能链接文件的目录
#     :param threshold: t test的阈值
#     :param fctype:功能连接的类型DFC或者SFC
#     :return: feature: 返回的特征，用来做SVM-RFE
#              feature_index: 返回的这些被筛选出来的特征的编号，以便之后进行生理解释
#     """
#     print('###加载文件')
#     file_dir_hc_test = find_file(dir_hc, ''.join(('_t_test_', str(threshold))))
#     file_dir_mdd_test = find_file(dir_mdd, ''.join(('_t_test_', str(threshold))))
#     # 三个返回值的存储地址
#     file_dir_hc_test_file = os.path.join(file_dir_hc_test, 'features')
#     file_dir_mdd_test_file = os.path.join(file_dir_mdd_test, 'features')
#     file_dir_feature_index = os.path.join(file_dir_hc_test, 'index')
#     # 如果已经做过了t test直接load
#     if os.path.exists(file_dir_hc_test_file) and os.path.exists(file_dir_mdd_test_file) and os.path.exists(
#             file_dir_feature_index):
#         feature_hc = scio.loadmat(file_dir_hc_test_file)['data']
#         feature_mdd = scio.loadmat(file_dir_mdd_test_file)['data']
#         feature_index = scio.loadmat(file_dir_feature_index)['data'][0]
#         ## 存的时候存成了二维，只有feature_index需要取[0]
#         print('###加载文件成功')
#     else:
#             # 如果没有做过t test进行下面的操作
#             # 目录转换
#             file_dir_hc = find_file(dir_hc, '_splice_along_time')
#             file_dir_hc = os.path.join(file_dir_hc, 't_test')
#             file_dir_mdd = find_file(dir_mdd, '_splice_along_time')
#             file_dir_mdd = os.path.join(file_dir_mdd, 't_test')
#             # 判断数据文件是否已经存在（是否已经进行过转换）
#             if os.path.exists(file_dir_hc) and os.path.exists(file_dir_mdd):
#                 pass
#             else:
#                 # 执行转换
#                 dfc2vector_parallel(dir_hc, fctype)
#                 dfc2vector_parallel(dir_mdd, fctype)
#             # load 矩阵
#             data_hc = scio.loadmat(file_dir_hc)['data']
#             data_mdd = scio.loadmat(file_dir_mdd)['data']
#             # 进行t_test得到每个特征的p值
#             print('###加载文件成功')
#             print('###进行t test')
#             p_value_temp = t_test_parallel(data_hc, data_mdd)
#             p_value_t =p_value_temp[:]
#             # print('hahahahha')
#             # print(p_value_t)
#             p_value = np.array(p_value_t)
#             # print('xixixixiix')
#             print(p_value)
#             print('### t test完成')
#             print('### 保存文件')
#             feature_index = np.array(range(0, p_value.size))
#             # 根据阈值对p_value进行筛选，得到特征对应的p值和编号
#             feature_hc = data_hc[:, p_value < threshold]
#             feature_mdd = data_mdd[:, p_value < threshold]
#             feature_index = feature_index[p_value < threshold]
#             # print('shapeshapeshape')
#             # print(feature_index[0])
#             # 保存在文件里，下次就可以直接load了，执行更快
#             if not os.path.exists(file_dir_hc_test):
#                 os.mkdir(file_dir_hc_test)
#             if not os.path.exists(file_dir_mdd_test):
#                 os.mkdir(file_dir_mdd_test)
#             scio.savemat(file_dir_hc_test_file, {'data': feature_hc})
#             scio.savemat(file_dir_mdd_test_file, {'data': feature_mdd})
#             scio.savemat(file_dir_feature_index, {'data': feature_index})
#     return feature_hc, feature_mdd, feature_index
# 根据编号，计算这个特征原来应该对应单个fc矩阵上半部分拉伸之后的哪一个，和生理拉伸矩阵关联可以得到生理区域（这里还不需要，先记下来编号）
# 加一个判断，如果数据文件存在就不用跑矩阵提取拼接，如果不存在就跑
# 加一个输入参数的可选项，是SVM还是LSTM就用不同数据文件中的数据，如果不存在就跑那些拉伸拼接的代码（可以放在新建的函数里面，这个函数就用来做t test)
# 返回的时候不仅返回选出来的特征，还要返回这些特征对应的编号，之后好研究是那两个区域的功能链接
# 要得到特征编号和生理区域的对应，先构造一个生理区域的连接矩阵，然后用相同的方法拉伸，就可以对应起来了，把这个得到的对应放到一个文件里存起来



    # SVM_data_prepare("/data/upload/duyongqi/mdd_prediction_brainnetcome/HC_DFC", "/data/upload/duyongqi/mdd_prediction_brainnetcome/MDD_DFC", 'DFC', vector)