# '''数据处理，希望把数据存成dataframe的格式，方便后续使用
# '''
# import anatomical
# import numpy as np 
# import scipy.io as scio
# import pandas as pd

# def data_prepare(filepath_matrix, index, column):
#     '''
#     把matrix转成dataframe
#     '''
#     pd.DataFrame(data=filepath_matrix, index =index, column=column )


# if __name__ == "__main__":
#     list_region = ['SFG_L_7_1', 'SFG_R_7_1', 'SFG_L_7_2', 'SFG_R_7_2', 'SFG_L_7_3', 'SFG_R_7_3', 'SFG_L_7_4', 'SFG_R_7_4', 'SFG_L_7_5', 'SFG_R_7_5', 'SFG_L_7_6', 'SFG_R_7_6', 'SFG_L_7_7', 'SFG_R_7_7', 'MFG_L_7_1', 'MFG_R_7_1', 'MFG_L_7_2', 'MFG_R_7_2', 'MFG_L_7_3', 'MFG_R_7_3', 'MFG_L_7_4', 'MFG_R_7_4', 'MFG_L_7_5', 'MFG_R_7_5', 'MFG_L_7_6', 'MFG_R_7_6', 'MFG_L_7_7', 'MFG_R_7_7', 'IFG_L_6_1', 'IFG_R_6_1', 'IFG_L_6_2', 'IFG_R_6_2', 'IFG_L_6_3', 'IFG_R_6_3', 'IFG_L_6_4', 'IFG_R_6_4', 'IFG_L_6_5', 'IFG_R_6_5', 'IFG_L_6_6', 'IFG_R_6_6', 'OrG_L_6_1', 'OrG_R_6_1', 'OrG_L_6_2', 'OrG_R_6_2', 'OrG_L_6_3', 'OrG_R_6_3', 'OrG_L_6_4', 'OrG_R_6_4', 'OrG_L_6_5', 'OrG_R_6_5', 'OrG_L_6_6', 'OrG_R_6_6', 'PrG_L_6_1', 'PrG_R_6_1', 'PrG_L_6_2', 'PrG_R_6_2', 'PrG_L_6_3', 'PrG_R_6_3', 'PrG_L_6_4', 'PrG_R_6_4', 'PrG_L_6_5', 'PrG_R_6_5', 'PrG_L_6_6', 'PrG_R_6_6', 'PCL_L_2_1', 'PCL_R_2_1', 'PCL_L_2_2', 'PCL_R_2_2', 'STG_L_6_1', 'STG_R_6_1', 'STG_L_6_2', 'STG_R_6_2', 'STG_L_6_3', 'STG_R_6_3', 'STG_L_6_4', 'STG_R_6_4', 'STG_L_6_5', 'STG_R_6_5', 'STG_L_6_6', 'STG_R_6_6', 'MTG_L_4_1', 'MTG_R_4_1', 'MTG_L_4_2', 'MTG_R_4_2', 'MTG_L_4_3', 'MTG_R_4_3', 'MTG_L_4_4', 'MTG_R_4_4', 'ITG_L_7_1', 'ITG_R_7_1', 'ITG_L_7_2', 'ITG_R_7_2', 'ITG_L_7_3', 'ITG_R_7_3', 'ITG_L_7_4', 'ITG_R_7_4', 'ITG_L_7_5', 'ITG_R_7_5', 'ITG_L_7_6', 'ITG_R_7_6', 'ITG_L_7_7', 'ITG_R_7_7', 'FuG_L_3_1', 'FuG_R_3_1', 'FuG_L_3_2', 'FuG_R_3_2', 'FuG_L_3_3', 'FuG_R_3_3', 'PhG_L_6_1', 'PhG_R_6_1', 'PhG_L_6_2', 'PhG_R_6_2', 'PhG_L_6_3', 'PhG_R_6_3', 'PhG_L_6_4', 'PhG_R_6_4', 'PhG_L_6_5', 'PhG_R_6_5', 'PhG_L_6_6', 'PhG_R_6_6', 'pSTS_L_2_1', 'pSTS_R_2_1', 'pSTS_L_2_2', 'pSTS_R_2_2', 'SPL_L_5_1', 'SPL_R_5_1', 'SPL_L_5_2', 'SPL_R_5_2', 'SPL_L_5_3', 'SPL_R_5_3', 'SPL_L_5_4', 'SPL_R_5_4', 'SPL_L_5_5', 'SPL_R_5_5', 'IPL_L_6_1', 'IPL_R_6_1', 'IPL_L_6_2', 'IPL_R_6_2', 'IPL_L_6_3', 'IPL_R_6_3', 'IPL_L_6_4', 'IPL_R_6_4', 'IPL_L_6_5', 'IPL_R_6_5', 'IPL_L_6_6', 'IPL_R_6_6', 'PCun_L_4_1', 'PCun_R_4_1', 'PCun_L_4_2', 'PCun_R_4_2', 'PCun_L_4_3', 'PCun_R_4_3', 'PCun_L_4_4', 'PCun_R_4_4', 'PoG_L_4_1', 'PoG_R_4_1', 'PoG_L_4_2', 'PoG_R_4_2', 'PoG_L_4_3', 'PoG_R_4_3', 'PoG_L_4_4', 'PoG_R_4_4', 'INS_L_6_1', 'INS_R_6_1', 'INS_L_6_2', 'INS_R_6_2', 'INS_L_6_3', 'INS_R_6_3', 'INS_L_6_4', 'INS_R_6_4', 'INS_L_6_5', 'INS_R_6_5', 'INS_L_6_6', 'INS_R_6_6', 'CG_L_7_1', 'CG_R_7_1', 'CG_L_7_2', 'CG_R_7_2', 'CG_L_7_3', 'CG_R_7_3', 'CG_L_7_4', 'CG_R_7_4', 'CG_L_7_5', 'CG_R_7_5', 'CG_L_7_6', 'CG_R_7_6', 'CG_L_7_7', 'CG_R_7_7', 'MVOcC _L_5_1', 'MVOcC _R_5_1', 'MVOcC _L_5_2', 'MVOcC _R_5_2', 'MVOcC _L_5_3', 'MVOcC _R_5_3', 'MVOcC _L_5_4', 'MVOcC _R_5_4', 'MVOcC _L_5_5', 'MVOcC _R_5_5', 'LOcC_L_4_1', 'LOcC_R_4_1', 'LOcC _L_4_2', 'LOcC _R_4_2', 'LOcC _L_4_3', 'LOcC _R_4_3', 'LOcC_L_4_4', 'LOcC_R_4_4', 'LOcC_L_2_1', 'LOcC_R_2_1', 'LOcC_L_2_2', 'LOcC_R_2_2', 'Amyg_L_2_1', 'Amyg_R_2_1', 'Amyg_L_2_2', 'Amyg_R_2_2', 'Hipp_L_2_1', 'Hipp_R_2_1', 'Hipp_L_2_2', 'Hipp_R_2_2', 'BG_L_6_1', 'BG_R_6_1', 'BG_L_6_2', 'BG_R_6_2', 'BG_L_6_3', 'BG_R_6_3', 'BG_L_6_4', 'BG_R_6_4', 'BG_L_6_5', 'BG_R_6_5', 'BG_L_6_6', 'BG_R_6_6', 'Tha_L_8_1', 'Tha_R_8_1', 'Tha_L_8_2', 'Tha_R_8_2', 'Tha_L_8_3', 'Tha_R_8_3', 'Tha_L_8_4', 'Tha_R_8_4', 'Tha_L_8_5', 'Tha_R_8_5', 'Tha_L_8_6', 'Tha_R_8_6', 'Tha_L_8_7', 'Tha_R_8_7', 'Tha_L_8_8', 'Tha_R_8_8']
#     #############
#     vector = anatomical.get_anatomical(list_region)
    
#     hc_splice = data_prepare()
import pandas as pd 
from sklearn.model_selection import train_test_split
# pd.DataFrame.join()
data = pd.read_pickle("/data/upload/duyongqi/mdd_prediction_brainnetcome/final_data_svm.pkl")
x = data.iloc[:,0:-1]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)
print(y_train)
x_train.insert(0, 'label', y_train)
x_test.insert(0, 'label', y_test)
print(x_train)
print(x_test)
x_train.to_csv('train.csv')
print('train done')
x_test.to_csv('test.csv')
print('test done')
# x_train.join(y_train).to_csv('C:/Users/荒荒儿/Pictures/train')
# x_test.join(x_train).to_csv('C:/Users/荒荒儿/Pictures/test')


