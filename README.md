# fmri data_based MDD classification
将预处理后的功能磁共振数据组织成相应的文件结构，自动实现数据准备、数据分割、模型训练和结果可视化。

# Content:
- [Usage](#Usage)
- [Ref-paper](#Ref_paper)

# Usage:
基于fmri数据的ml&amp;dl方法疾病分类
通过将预处理后的数据组织到相应的网格文件结构中，实现数据的集成、数据的自动编制、数据的自动划分、模型的训练和模型的可视化绘制

# 原理：
- SFC数据准备
![SFC](https://github.com/duyongqi/fmri-data_based-major-depressive-disorder-ML-DL-classification-/blob/main/image/SFC_data_preparation.jpg)
- DFC数据准备
![DFC](https://github.com/duyongqi/fmri-data_based-major-depressive-disorder-ML-DL-classification-/blob/main/image/DFC_data_preparation.jpg)
- 分类器
![SVM](https://github.com/duyongqi/fmri-data_based-major-depressive-disorder-ML-DL-classification-/blob/main/image/model_pipeline.jpg)
- 选择特征的生理解释
![anatomical](https://github.com/duyongqi/fmri-data_based-major-depressive-disorder-ML-DL-classification-/blob/main/image/anatomical_1.jpg)
![anatomical](https://github.com/duyongqi/fmri-data_based-major-depressive-disorder-ML-DL-classification-/blob/main/image/anatomical_2.jpg)
- 选择特征的可视化
用[BrainNet Viewer](https://www.nitrc.org/projects/bnv/)工具进行选择出来的特征的可视化
- 代码框架
![pipeline](https://github.com/duyongqi/fmri-data_based-major-depressive-disorder-ML-DL-classification-/blob/main/image/pipeline.svg)

## 输入文件格式：
对原始fMRI数据用[DPABI](http://rfmri.org/dpabi)工具进行处理后（包括脑区分割），用[GRENTA](https://www.frontiersin.org/articles/10.3389/fnhum.2015.00386/full)工具进行SFC和DFC的提取，得到每个被试的SFC和DFC矩阵,为index.mat文件格式，最终输入文件格式应该如下：(SFC中的mat文件是2维的，DFC中的mat文件是3维的，有一个维度是时间)
<details><summary>文件结构</summary>
<p>

--SFC

----HC

------0001.mat

------0002.mat

------0003.mat

------...

----MDD

------0001.mat

------0002.mat

------0003.mat

------...

--DFC

----HC

------0001.mat

------0002.mat

------0003.mat

------...

----MDD

------0001.mat

------0002.mat

------0003.mat

------...

</p>
</details>

## 代码结构:
- anatomical.py

做特征选择之后，对选择出来的特征进行生理解释，也就是进行区域的对应，找出选择出来的特征是哪些区域之间的功能连接)

- draw.py 

训练和测试结束之后，绘制gridsearch的过程，绘制AUC 

- **main.py**  

主函数，输入HC和MDD文件的路径，FC类别，模型种类(SVM/LSTM);
进行自动特征整合，t test特征选择，SVM-RFE特征选择，SVM分类, Gridsearch寻找最佳参数，以及特征的生理解释的自动对应输出，测试结果的AUC图像的自动绘制。

## 代码用法:

```python
python main.py SVM sfc/dfc "hc_dir" "mdd_dir" --threshold 0.2 --atlas AAL
```

<details><summary>代码输入参数说明</summary>
<p>

```python
usage: main.py [-h] [--threshold THRESHOLD] [--atlas ATLAS] {SVM,LSTM,oLSTM} {DFC,SFC} hc mdd 

预处理之后数据的抑郁症诊断，可以选择三种方法，一种是DFC+特征选择SVM，一种是DFC+LSTM，一种是直接LSTM

positional arguments:
  {SVM,LSTM,oLSTM}      分类方法类别
  {DFC,SFC}             功能连接类别
  hc                    正常组FC目录
  mdd                   MDD组FC目录

optional arguments:
  -h, --help            帮助
  --threshold THRESHOLD, -t THRESHOLD
                        t test的阈值
  --atlas ATLAS, -a ATLAS
                        选择使用的分割图，注意要和输入的对应的地址中数据使用的地址一致，默认是AAL90
```

</p>
</details>

# 中间文件
- 
# 注意事项：
1. main.py中实现了多进程，用了15核的CPU，在运行之前请将其更改为合适的个数，代码在

```python
p = Pool(processes=15)
```
2. LSTM方法的实现还未上传
3. 输入文件目录结构一定要按照上文描述组织

# Ref_paper:
- [Paper_ref](https://www.frontiersin.org/articles/10.3389/fninf.2020.00025/full)[^1]


[^1]: Castellazzi, Gloria, et al. "A machine learning approach for the differential diagnosis of Alzheimer and Vascular Dementia Fed by MRI selected features." Frontiers in neuroinformatics 14 (2020): 25.

### IF THIS IS USEFUL FOR YOU, STAR FOR ME PLS! THKS!! :satisfied: