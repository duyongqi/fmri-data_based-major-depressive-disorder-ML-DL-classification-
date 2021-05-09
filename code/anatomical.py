'''
根据index打印生理标识文件
'''
import numpy as np
import random
import string


def list2matrix(str1, str2):
    """
    字符数组连接成矩阵，矩阵的值是数组的值用_连接，用来得到脑区连接的对应关系
    :param str1: 第一个字符数组
    :param str2: 第二个字符数组
    :return:
    """
    matrix_out = []
    for i in range(len(str1)):
        row = []
        for j in range(len(str2)):
            row.append('_'.join((str1[i], str2[j])))
        matrix_out.append(row)
    matrix_out = np.array(matrix_out)
    return matrix_out


def matrix2vector(mat):
    """
    矩阵上半部分提取加拉伸
    :param mat:
    :return:
    """
    vector = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[1]):
            vector.append(mat[i][j])
    return vector


def print_anatomical(index_vector, vector):
    """
    根据index输出对应的字符
    :param index_vector:
    :param vector:
    :return:
    """
    for i in index_vector:
        print(vector[i])

def print_anatomical_unique(index_vector, vector):
    """
    根据index输出对应的字符
    :param index_vector:
    :param vector:
    :return:
    """
    unique_index = np.unique(index_vector)
    for i in unique_index:
        print(vector[i])
    print(unique_index.size)


def get_anatomical(list_region):
    """
    输入脑区（按顺序），输出矩阵上半部分拉伸后的对应区域，和fc矩阵的拉伸方法一致，保证拉伸前后对应的区域解释一致
    :param list_region:
    :return:
    """
    mat = list2matrix(list_region, list_region)
    vector = matrix2vector(mat)
    return vector

# 等待添加功能：用来分隔的脑区可选,需要做很多图对应的dfc,然后之后传地址的时候传不同的地址，因为本代码并没有计算DFC的功能
def anatomical_vector_generator(atlas):
    if atlas == 'AAL90':
        list_region = [
        '1 Precentral_L 2001', 
        '2 Precentral_R 2002', 
        '3 Frontal_Sup_L 2101', 
        '4 Frontal_Sup_R 2102', 
        '5 Frontal_Sup_Orb_L 2111', 
        '6 Frontal_Sup_Orb_R 2112', 
        '7 Frontal_Mid_L 2201', 
        '8 Frontal_Mid_R 2202', 
        '9 Frontal_Mid_Orb_L 2211', 
        '10 Frontal_Mid_Orb_R 2212', 
        '11 Frontal_Inf_Oper_L 2301', 
        '12 Frontal_Inf_Oper_R 2302', 
        '13 Frontal_Inf_Tri_L 2311', 
        '14 Frontal_Inf_Tri_R 2312', 
        '15 Frontal_Inf_Orb_L 2321', 
        '16 Frontal_Inf_Orb_R 2322', 
        '17 Rolandic_Oper_L 2331', 
        '18 Rolandic_Oper_R 2332', 
        '19 Supp_Motor_Area_L 2401', 
        '20 Supp_Motor_Area_R 2402', 
        '21 Olfactory_L 2501', 
        '22 Olfactory_R 2502', 
        '23 Frontal_Sup_Medial_L 2601', 
        '24 Frontal_Sup_Medial_R 2602', 
        '25 Frontal_Med_Orb_L 2611', 
        '26 Frontal_Med_Orb_R 2612', 
        '27 Rectus_L 2701', 
        '28 Rectus_R 2702', 
        '29 Insula_L 3001', 
        '30 Insula_R 3002', 
        '31 Cingulum_Ant_L 4001', 
        '32 Cingulum_Ant_R 4002', 
        '33 Cingulum_Mid_L 4011', 
        '34 Cingulum_Mid_R 4012', 
        '35 Cingulum_Post_L 4021', 
        '36 Cingulum_Post_R 4022', 
        '37 Hippocampus_L 4101', 
        '38 Hippocampus_R 4102', 
        '39 ParaHippocampal_L 4111', 
        '40 ParaHippocampal_R 4112', 
        '41 Amygdala_L 4201', 
        '42 Amygdala_R 4202', 
        '43 Calcarine_L 5001', 
        '44 Calcarine_R 5002', 
        '45 Cuneus_L 5011', 
        '46 Cuneus_R 5012', 
        '47 Lingual_L 5021', 
        '48 Lingual_R 5022', 
        '49 Occipital_Sup_L 5101', 
        '50 Occipital_Sup_R 5102', 
        '51 Occipital_Mid_L 5201', 
        '52 Occipital_Mid_R 5202', 
        '53 Occipital_Inf_L 5301', 
        '54 Occipital_Inf_R 5302', 
        '55 Fusiform_L 5401', 
        '56 Fusiform_R 5402', 
        '57 Postcentral_L 6001', 
        '58 Postcentral_R 6002', 
        '59 Parietal_Sup_L 6101', 
        '60 Parietal_Sup_R 6102', 
        '61 Parietal_Inf_L 6201', 
        '62 Parietal_Inf_R 6202', 
        '63 SupraMarginal_L 6211', 
        '64 SupraMarginal_R 6212', 
        '65 Angular_L 6221', 
        '66 Angular_R 6222', 
        '67 Precuneus_L 6301', 
        '68 Precuneus_R 6302', 
        '69 Paracentral_Lobule_L 6401', 
        '70 Paracentral_Lobule_R 6402', 
        '71 Caudate_L 7001', 
        '72 Caudate_R 7002', 
        '73 Putamen_L 7011', 
        '74 Putamen_R 7012', 
        '75 Pallidum_L 7021', 
        '76 Pallidum_R 7022', 
        '77 Thalamus_L 7101', 
        '78 Thalamus_R 7102', 
        '79 Heschl_L 8101', 
        '80 Heschl_R 8102', 
        '81 Temporal_Sup_L 8111', 
        '82 Temporal_Sup_R 8112', 
        '83 Temporal_Pole_Sup_L 8121', 
        '84 Temporal_Pole_Sup_R 8122', 
        '85 Temporal_Mid_L 8201', 
        '86 Temporal_Mid_R 8202', 
        '87 Temporal_Pole_Mid_L 8211', 
        '88 Temporal_Pole_Mid_R 8212', 
        '89 Temporal_Inf_L 8301', 
        '90 Temporal_Inf_R 8302', 
        '91 Cerebelum_Crus1_L 9001', 
        '92 Cerebelum_Crus1_R 9002', 
        '93 Cerebelum_Crus2_L 9011', 
        '94 Cerebelum_Crus2_R 9012', 
        '95 Cerebelum_3_L 9021', 
        '96 Cerebelum_3_R 9022', 
        '97 Cerebelum_4_5_L 9031', 
        '98 Cerebelum_4_5_R 9032', 
        '99 Cerebelum_6_L 9041', 
        '100 Cerebelum_6_R 9042', 
        '101 Cerebelum_7b_L 9051', 
        '102 Cerebelum_7b_R 9052', 
        '103 Cerebelum_8_L 9061', 
        '104 Cerebelum_8_R 9062', 
        '105 Cerebelum_9_L 9071', 
        '106 Cerebelum_9_R 9072', 
        '107 Cerebelum_10_L 9081', 
        '108 Cerebelum_10_R 9082', 
        '109 Vermis_1_2 9100', 
        '110 Vermis_3 9110', 
        '111 Vermis_4_5 9120', 
        '112 Vermis_6 9130', 
        '113 Vermis_7 9140', 
        '114 Vermis_8 9150', 
        '115 Vermis_9 9160', 
        '116 Vermis_10 9170']
        #############
        vector = get_anatomical(list_region)
    elif atlas == 'AAL100':
        list_region = ['aal100']
        #############
        vector = anatomical.get_anatomical(list_region)
    elif atlas == 'brainnetome':
        list_region = ['SFG_L_7_1', 'SFG_R_7_1', 'SFG_L_7_2', 'SFG_R_7_2', 'SFG_L_7_3', 'SFG_R_7_3', 'SFG_L_7_4', 'SFG_R_7_4', 'SFG_L_7_5', 'SFG_R_7_5', 'SFG_L_7_6', 'SFG_R_7_6', 'SFG_L_7_7', 'SFG_R_7_7', 'MFG_L_7_1', 'MFG_R_7_1', 'MFG_L_7_2', 'MFG_R_7_2', 'MFG_L_7_3', 'MFG_R_7_3', 'MFG_L_7_4', 'MFG_R_7_4', 'MFG_L_7_5', 'MFG_R_7_5', 'MFG_L_7_6', 'MFG_R_7_6', 'MFG_L_7_7', 'MFG_R_7_7', 'IFG_L_6_1', 'IFG_R_6_1', 'IFG_L_6_2', 'IFG_R_6_2', 'IFG_L_6_3', 'IFG_R_6_3', 'IFG_L_6_4', 'IFG_R_6_4', 'IFG_L_6_5', 'IFG_R_6_5', 'IFG_L_6_6', 'IFG_R_6_6', 'OrG_L_6_1', 'OrG_R_6_1', 'OrG_L_6_2', 'OrG_R_6_2', 'OrG_L_6_3', 'OrG_R_6_3', 'OrG_L_6_4', 'OrG_R_6_4', 'OrG_L_6_5', 'OrG_R_6_5', 'OrG_L_6_6', 'OrG_R_6_6', 'PrG_L_6_1', 'PrG_R_6_1', 'PrG_L_6_2', 'PrG_R_6_2', 'PrG_L_6_3', 'PrG_R_6_3', 'PrG_L_6_4', 'PrG_R_6_4', 'PrG_L_6_5', 'PrG_R_6_5', 'PrG_L_6_6', 'PrG_R_6_6', 'PCL_L_2_1', 'PCL_R_2_1', 'PCL_L_2_2', 'PCL_R_2_2', 'STG_L_6_1', 'STG_R_6_1', 'STG_L_6_2', 'STG_R_6_2', 'STG_L_6_3', 'STG_R_6_3', 'STG_L_6_4', 'STG_R_6_4', 'STG_L_6_5', 'STG_R_6_5', 'STG_L_6_6', 'STG_R_6_6', 'MTG_L_4_1', 'MTG_R_4_1', 'MTG_L_4_2', 'MTG_R_4_2', 'MTG_L_4_3', 'MTG_R_4_3', 'MTG_L_4_4', 'MTG_R_4_4', 'ITG_L_7_1', 'ITG_R_7_1', 'ITG_L_7_2', 'ITG_R_7_2', 'ITG_L_7_3', 'ITG_R_7_3', 'ITG_L_7_4', 'ITG_R_7_4', 'ITG_L_7_5', 'ITG_R_7_5', 'ITG_L_7_6', 'ITG_R_7_6', 'ITG_L_7_7', 'ITG_R_7_7', 'FuG_L_3_1', 'FuG_R_3_1', 'FuG_L_3_2', 'FuG_R_3_2', 'FuG_L_3_3', 'FuG_R_3_3', 'PhG_L_6_1', 'PhG_R_6_1', 'PhG_L_6_2', 'PhG_R_6_2', 'PhG_L_6_3', 'PhG_R_6_3', 'PhG_L_6_4', 'PhG_R_6_4', 'PhG_L_6_5', 'PhG_R_6_5', 'PhG_L_6_6', 'PhG_R_6_6', 'pSTS_L_2_1', 'pSTS_R_2_1', 'pSTS_L_2_2', 'pSTS_R_2_2', 'SPL_L_5_1', 'SPL_R_5_1', 'SPL_L_5_2', 'SPL_R_5_2', 'SPL_L_5_3', 'SPL_R_5_3', 'SPL_L_5_4', 'SPL_R_5_4', 'SPL_L_5_5', 'SPL_R_5_5', 'IPL_L_6_1', 'IPL_R_6_1', 'IPL_L_6_2', 'IPL_R_6_2', 'IPL_L_6_3', 'IPL_R_6_3', 'IPL_L_6_4', 'IPL_R_6_4', 'IPL_L_6_5', 'IPL_R_6_5', 'IPL_L_6_6', 'IPL_R_6_6', 'PCun_L_4_1', 'PCun_R_4_1', 'PCun_L_4_2', 'PCun_R_4_2', 'PCun_L_4_3', 'PCun_R_4_3', 'PCun_L_4_4', 'PCun_R_4_4', 'PoG_L_4_1', 'PoG_R_4_1', 'PoG_L_4_2', 'PoG_R_4_2', 'PoG_L_4_3', 'PoG_R_4_3', 'PoG_L_4_4', 'PoG_R_4_4', 'INS_L_6_1', 'INS_R_6_1', 'INS_L_6_2', 'INS_R_6_2', 'INS_L_6_3', 'INS_R_6_3', 'INS_L_6_4', 'INS_R_6_4', 'INS_L_6_5', 'INS_R_6_5', 'INS_L_6_6', 'INS_R_6_6', 'CG_L_7_1', 'CG_R_7_1', 'CG_L_7_2', 'CG_R_7_2', 'CG_L_7_3', 'CG_R_7_3', 'CG_L_7_4', 'CG_R_7_4', 'CG_L_7_5', 'CG_R_7_5', 'CG_L_7_6', 'CG_R_7_6', 'CG_L_7_7', 'CG_R_7_7', 'MVOcC _L_5_1', 'MVOcC _R_5_1', 'MVOcC _L_5_2', 'MVOcC _R_5_2', 'MVOcC _L_5_3', 'MVOcC _R_5_3', 'MVOcC _L_5_4', 'MVOcC _R_5_4', 'MVOcC _L_5_5', 'MVOcC _R_5_5', 'LOcC_L_4_1', 'LOcC_R_4_1', 'LOcC _L_4_2', 'LOcC _R_4_2', 'LOcC _L_4_3', 'LOcC _R_4_3', 'LOcC_L_4_4', 'LOcC_R_4_4', 'LOcC_L_2_1', 'LOcC_R_2_1', 'LOcC_L_2_2', 'LOcC_R_2_2', 'Amyg_L_2_1', 'Amyg_R_2_1', 'Amyg_L_2_2', 'Amyg_R_2_2', 'Hipp_L_2_1', 'Hipp_R_2_1', 'Hipp_L_2_2', 'Hipp_R_2_2', 'BG_L_6_1', 'BG_R_6_1', 'BG_L_6_2', 'BG_R_6_2', 'BG_L_6_3', 'BG_R_6_3', 'BG_L_6_4', 'BG_R_6_4', 'BG_L_6_5', 'BG_R_6_5', 'BG_L_6_6', 'BG_R_6_6', 'Tha_L_8_1', 'Tha_R_8_1', 'Tha_L_8_2', 'Tha_R_8_2', 'Tha_L_8_3', 'Tha_R_8_3', 'Tha_L_8_4', 'Tha_R_8_4', 'Tha_L_8_5', 'Tha_R_8_5', 'Tha_L_8_6', 'Tha_R_8_6', 'Tha_L_8_7', 'Tha_R_8_7', 'Tha_L_8_8', 'Tha_R_8_8']
        #############
        vector = anatomical.get_anatomical(list_region)
    else:
        vector = []
    return vector
if __name__ == "__main__":
    # 做index和解剖区域对应的一维序列
    anatomical = np.empty(200, dtype=np.object)
    for i in range(200):
        anatomical[i] = ''.join(random.sample(string.ascii_letters + string.digits, 10))

    # 转成相关矩阵
    a = np.array(list2matrix(anatomical, anatomical))
    # 提取字符矩阵上三角
    b = matrix2vector(a)
    # index
    print_anatomical([0, 2, 199], b)
