# fmri data_based MDD classification
By organizing the pre-processed fmri data into the corresponding file structure,data preparation, data split, model training and result visualization will be implemented automatically.
# Content
- [Usage](##Usage)
- [Ref-paper](##Ref_paper)
# Usage
a fmri_data based disease classification  using ML&amp;DL methods
By organizing the pre-processed data into the corresponding grid file structure, it can be integrated, automated data preparation, data division, model training and visual drawing of model 
## 输入文件格式：
对原始fMRI数据用DPABI工具进行处理后（包括脑区分割），用GRENTA工具进行SFC和DFC的提取，得到每个被试的SFC和DFC矩阵,为index.mat文件格式，最终输入文件格式应该如下：(SFC中的mat文件是2维的，DFC中的mat文件是3维的，有一个维度是时间)
<details><summary>CLICK ME</summary>
<p>

--SFC

--HC

  --0001.mat

  --0002.mat

  --0003.mat

  ...

--MDD

  --0001.mat

  --0002.mat

  --0003.mat

  ...

--DFC

--HC

  --0001.mat

  --0002.mat

  --0003.mat

  ...

--MDD

  --0001.mat

  --0002.mat

  --0003.mat

  ...

</p>
</details>

## 代码结构
- anatomical.py

做特征选择之后，对选择出来的特征进行生理解释，也就是进行区域的对应，找出选择出来的特征是哪些区域之间的功能连接)

- draw.py 

训练和测试结束之后，绘制gridsearch的过程，绘制AUC 

- **main.py**  

主函数，输入HC和MDD文件的路径，FC类别，模型种类(SVM/LSTM);
进行自动特征整合，t test特征选择，SVM-RFE特征选择，SVM分类, Gridsearch寻找最佳参数，以及特征的生理解释的自动对应输出，测试结果的AUC图像的自动绘制。

## 代码用法

```
python main.py SVM sfc/dfc "hc_dir" "mdd_dir" --threshold 0.2 --atlas AAL
```

```
usage: main.py [-h] [--threshold THRESHOLD] [--atlas ATLAS] {SVM,LSTM,oLSTM} {DFC,SFC} hc mdd 

预处理之后数据的抑郁症诊断，可以选择三种方法，一种是DFC+特征选择SVM，一种是DFC+LSTM，一种是直接LSTM

positional arguments:
  {SVM,LSTM,oLSTM}      分类方法类别
  {DFC,SFC}             功能连接类别
  hc                    正常组FC目录
  mdd                   MDD组FC目录

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD, -t THRESHOLD
                        t test的阈值
  --atlas ATLAS, -a ATLAS
                        选择使用的分割图，注意要和输入的对应的地址中数据使用的地址一致，默认是AAL90
```

# Ref_paper
- [Paper_ref](https://www.frontiersin.org/articles/10.3389/fninf.2020.00025/full)[^1]

### IF THIS IS USEFUL FOR YOU, STAR FOR ME PLS! THKS!! :satisfied:

[^1]: 
Castellazzi, Gloria, et al. "A machine learning approach for the differential diagnosis of Alzheimer and Vascular Dementia Fed by MRI selected features." Frontiers in neuroinformatics 14 (2020): 25.