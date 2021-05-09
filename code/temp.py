'''
多进程测试文件
'''
import numpy as np
import time
from multiprocessing import Process, cpu_count, Pool, Manager
import multiprocessing as mul
import anatomical
from scipy import stats
import scipy.io as scio
import numba as nb

# # @nb.jit(nopython=True)
# def t_test(matrix_one, matrix_two, start, length, p_value):
#     # print('hahahahah')
#     # p_value[i] = stats.ttest_ind(matrix_one, matrix_two).pvalue
#     for j in range(length):
#         a = stats.ttest_ind(a[:, j], b[:, j]).pvalue
#         print(a)
#         p_value[start + j] = a
#         # print(p_value)

def fun(a):
    return a+1
if __name__ == "__main__":
    # # print('cpu核{}'.format(cpu_count()))
    # a = scio.loadmat("/home/duyongqi/mddprediction_fmri/HC_splice_along_time/t_test")['data']
    # b = scio.loadmat("/home/duyongqi/mddprediction_fmri/MDD_splice_along_time/t_test")['data']
    # # a = np.array([range(1,1001),range(21,1021),range(21,1021),range(31,1031),range(21,1021),range(81,1081),range(21,1021)])
    # # b = np.array([range(8,1008)]*10)
    # manager = Manager()
    # p_value = np.empty(a.shape[1])
    # p_value1 = manager.Array('f', sequence=np.empty(a.shape[1]))

    # # t1 = time.time()
    # # for i in range(1000):
    # #     # time.sleep(0.0001)
    # #     # dd= stats.ttest_ind(a[:, i], b[:, i]).pvalue
    # #     p_value[i] = stats.ttest_ind(a[:, i], b[:, i]).pvalue
    # # t2 = time.time()
    # # print(p_value)
    # # print('运行时间{}'.format(str(t2 - t1)))
    # mm = stats.ttest_ind(a[:, 1], b[:, 1]).pvalue
    # print(mm)

    # t11 = time.time()
    # p = Pool(processes=20)
    # length = a.shape[1]//50
    # for i in range(50):
    #     start = i*length
    #     print(start)
    #     end = min(start + length, a.shape[1])
    #     p.apply_async(t_test ,(a[:, start:end], b[:, start:end], start, length, p_value1))
    #     # print(a)
    # p.close()
    # p.join()
    # # print(p_value1[1])
    # t22 = time.time()
    # print('运行时间{}'.format(str(t22 - t11)))


    # # p = Pool(processes=4)
    # # a = [1, 2, 3, 3, 5, 3, 3 , 2, 4, 2, 4, 4, 5, 7,5,1,1,8,5,8,5,6,2,8,9,7,6,2,3,6,5,8,9]
    # # # mul.Array(a ,[1, 2, 3])
    # # for i in range(10):
    # #     index = p.apply_async(f, (a,i))
    # #     a[i] = index.get()
    # # p.close()
    # # p.join()
    # # print(a)
    # # p.join()
    # # print(b)




    # # 已经做过的check:
    # # 检查了splice之后的文件的每一行的第一个和第二个是不是和每个人的原始矩阵中的第一行的第二个和第三个对应
    # # 即：验证拉伸这一步的有效应
    # # 等待：t test检查，单独拿出来一列试试
    # p = Pool(20)
    # results = []
    # for i in range(100):
    #     a = i * 2
    #     b = p.apply_async(fun, (a,))
    #     results.append(b)
    # p.close()
    # p.join()
    # for i in results:
    #     print(i.get())

    # file = open("/data/upload/duyongqi/mdd/HC_SFC/GretnaSFCMatrixZ/zHC001.txt", encoding='utf-8')
    # a = file.readlines()
    # b = np.loadtxt("/data/upload/duyongqi/mdd/HC_SFC/GretnaSFCMatrixZ/zHC001.txt", encoding='utf-8')
    # c = 1




