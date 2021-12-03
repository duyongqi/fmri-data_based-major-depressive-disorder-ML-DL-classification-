# fmri data_based MDD classification

将预处理后的功能磁共振数据组织成相应的文件结构，自动实现数据准备、数据分割、模型训练和结果可视化。

注意在过程中为了防止程序中断需要重新运行程序花费的时间，在运行过程中会自动生成中间文件，这样即使中断了，上一步的处理结果也会以文件的形式保存下来，中间文件包括HC/MDD_splice_along_time, model等，参见[中间文件](##中间文件)  


# Content:
- [Usage](#Usage)
  - [输入文件格式](##输入文件格式)
  - [代码格式](##代码结构)
  - [代码用法](##代码用法)  
  - [中间文件](##中间文件)   
- [原理](#原理)
- [注意事项](#注意事项)

- [Ref-paper](#Ref_paper)

# Usage:
基于fmri数据的ml&amp;dl方法疾病分类
通过将预处理后的数据组织到相应的网格文件结构中，实现数据的集成、数据的自动编制、数据的自动划分、模型的训练和模型的可视化绘制

## 输入文件格式：
对原始fMRI数据用[DPABI](http://rfmri.org/dpabi)工具进行处理后（包括脑区分割）；

用[GRENTA](https://www.frontiersin.org/articles/10.3389/fnhum.2015.00386/full)工具进行SFC和DFC的提取；

得到每个被试的SFC和DFC矩阵,为index.mat文件格式，最终输入文件格式应该如下：(SFC中的mat文件是2维的，DFC中的mat文件是3维的，有一个维度是时间)

<details><summary>文件结构</summary>
<p>

参考本repo中的文件格式，有一些文件夹是生成的中间文件，最初始的输入文件格式是下面这样，主要是将SFC和DFC分开放，SFC/DFC中的HC和MDD分开放，为了能容下中间文件的更好的查看方式，注意HC_Data才是存放HC数据的地方，而不是HC；HC是存放HC这一类的总目录（包括HC数据和生成的中间文件）。
- --SFC_Data

----HC

------HC_Data

--------0001.mat

--------0002.mat

--------0003.mat


----MDD

------MDD_Data

--------0001.mat

--------0002.mat

--------0003.mat


- --DFC_Data

----HC

------HC_Data

--------0001.mat

--------0002.mat

--------0003.mat


----MDD

------MDD_Data

--------0001.mat

--------0002.mat

--------0003.mat

</p>
</details>

## 代码结构:
<!-- ```diff -->

- anatomical.py

做特征选择之后，对选择出来的特征进行生理解释，也就是进行区域的对应，找出选择出来的特征是哪些区域之间的功能连接)

- draw.py 

训练和测试结束之后，绘制gridsearch的过程，绘制AUC 

- svm.py 

分类模型pipeline

- **main.py**  

主函数，输入HC和MDD文件的路径，FC类别，模型种类(SVM/LSTM);
进行自动特征整合，t test特征选择，SVM-RFE特征选择，SVM分类, Gridsearch寻找最佳参数，以及特征的生理解释的自动对应输出，测试结果的AUC图像的自动绘制。

<!-- ``` -->

## 代码用法:

```python
python main.py SVM sfc "SFC_Data\HC\HC_Data" "SFC_Data\MDD\MDD_Data" --threshold 0.2 --atlas AAL
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

## 中间文件
<!-- ```diff -->

参考[原理](#原理)理解

- HC/MDD_splice_alone_time

一类人的SFC/DFC矩阵拉直之后堆叠成的矩阵

  - SFC

直接拉伸成一维，所有人的堆叠，得到二维矩阵

  - DFC

对每一个人，将DFC的每一个矩阵拉伸成一维，将所有的一维向量拼接在一起；接着将所有人的拉伸拼接后的向量堆叠在一起，得到一个矩阵，一个维度是人的编号，一个维度是FC特征

- HC/MDD_stack_alone_time

一类人的DFC矩阵，每一个人的DFC中的每一个FC拉伸之后，将他们进行堆叠成一个二维矩阵，一个维度是时间；接着将所有人的矩阵堆叠在一起得到一个三维矩阵，一个维度是人编号，一个维度是时间，还有一个维度是拉伸后的FC

- model

存储训练后的模型，之后在进行生理解释和可视化的时候，可以直接获取训练好的模型进行后续运算

- SFC

SFC相关结果图

- DFC

DFC相关结果图


<!-- ``` -->

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


# 注意事项：
1. main.py中实现了多进程，用了15核的CPU，在运行之前请将其更改为合适的个数，代码在

```python
p = Pool(processes=15)
# 把15改成你的计算机的cpu内核数
```
2. LSTM方法的实现还未上传
3. 输入文件目录结构一定要按照上文描述组织

# Ref_paper:
- [Paper_ref](https://www.frontiersin.org/articles/10.3389/fninf.2020.00025/full)[^1]


[^1]: Castellazzi, Gloria, et al. "A machine learning approach for the differential diagnosis of Alzheimer and Vascular Dementia Fed by MRI selected features." Frontiers in neuroinformatics 14 (2020): 25.

### IF THIS IS USEFUL FOR YOU, STAR FOR ME PLS! THKS!! :satisfied: