# 数据处理相关

#### 数据分析常规流程一般包括（A typical data science process might look like this）:

* Project Scoping / Data Collection  数据收集
* Exploratory Analysis  数据探索
* Data Cleaning  数据清洗 
* Feature Engineering  特征工程
* Model Training \(including cross-validation to tune hyper-parameters\)  模型训练，调参
* Project Delivery / Insights  结果解读

#### EDA：

        In statistics, exploratory data analysis\(EDA\) is an approach to analyzing data sets to summarize their maincharacteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. Exploratory data analysis was promoted by John Tukey to encourage statisticians to explore the data, and possibly formulate hypotheses that could lead to new data collection and experiments. EDA is different from initial data analysis \(IDA\), which focuses more narrowly on checking assumptions required for model fitting and hypothesistesting, and handling missing values and making transformations of variables as needed. EDA encompasses IDA.

步骤： 

1. Form hypotheses/develop investigation theme to explore 
2. Wrangle data清理数据 [datawrangler](http://vis.stanford.edu/wrangler/) 
3. Assess quality of data评价数据质量
4. Profile data数据报表
5. Explore each individual variable in the dataset探索分析每个变量
6. Assess the relationship between each variable and the target探索每个自变量与因变量之间的关系
7. Assess interactions between variables探索每个自变量之间的相关性
8. Explore data across many dimensions从不同的维度来分析数据

探索拓展：

1. 写出一系列你自己的假设，然后做更深入的数据分析
2. 记录下自己探索过程中更进一步的数据分析过程
3. 把自己的中间的结果给自己的同行看看，让他们能够给你一些更有拓展性的反馈、或者意见。
4. 将可视化与结果结合一起。探索性数据分析，就是依赖你好的模型意识，（在《深入浅出数据分析》$$p_{34}$$中，把模型的敏感度叫心智模型，最初的心智模型可能错了，一旦自己的结果违背自己的假设，就要立即回去详细的思考）。所以我们在数据探索的尽可能把自己的可视化图和结果放一起，这样便于进一步分析。

常见的数据预处理方法： 

1. 取对数log：当数据的峰值很高，通过将数据取对数能够将数据归一化处理。能够放大差异。
2. 连续变量分组\(bin\)，分组连续变量，能够更加简便的了解观测值的分布。
3. **简化类别：一个单一的数据，往往类别太多会让人迷乱，一般不想超过8-10列，那就尽量找到重要的类别。（机器学习里面这一个部分很重要，和特征选择一样）**

探索分析步骤： 

* 探索变量
* 简单的描述统计，类别，缺失值，均值，类型等
* 数据切分\(透视表\) 例子：

  1. 横截面：一个时期内所有国家的数据
  2. 时间序列：一个国家随着时间推移的数据
  3. 面板数据：所有国家随着时间的推移数据
  4. 地理空间：所有地理上相互关联的数据

* 数据质量评估

  1. 评估缺失值数据，系统还是随机
  2. 标签包含给定字段丢失数据的默认值
  3. 确定质量评估抽样策略和初始EDA
  4. 时间数据类型，保证格式的一致性和粒度的数据，并执行对数据的所有日期的检查
  5. 查看每个字段数据类型
  6. 对于离散值类型，确保数据格式一致，评估不同值和唯一百分比的数据，并对答案的类型进行正确检查
  7. 连续数据类型，进行描述性统计，并对值进行检查
  8. 常用工具包[missingno](https://github.com/ResidentMario/missingno)、[pivottablejs](https://github.com/nicolaskruchten/jupyter_pivottablejs)、pandas\_profiling
  9. 数据峰度和偏度的研究

峰度$$Kurtosis = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\overline x)^4/SD^4-3$$

> 峰度是描述总体中所有取值分布形态陡缓程度的统计量。这个统计量需要与正态分布相比较，峰度为0表示该总体数据分布与正态分布的陡缓程度相同；峰度大于0表示该总体数据分布与正态分布相比较为陡峭，为尖顶峰；峰度小于0表示该总体数据分布与正态分布相比较为平坦，为平顶峰。峰度的绝对值数值越大表示其分布形态的陡缓程度与正态分布的差异程度越大。

偏度$$Skewness = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\overline x)^3/SD^3$$

> 偏度与峰度类似，它也是描述数据分布形态的统计量，其描述的是某总体取值分布的对称性。这个统计量同样需要与正态分布相比较，偏度为0表示其数据分布形态与正态分布的偏斜程度相同；偏度大于0表示其数据分布形态与正态分布相比为正偏或右偏，即有一条长尾巴拖在右边，数据右端有较多的极端值；偏度小于0表示其数据分布形态与正态分布相比为负偏或左偏，即有一条长尾拖在左边，数据左端有较多的极端值。偏度的绝对值数值越大表示其分布形态的偏斜程度越大。

* 可视化，变量之间的相关性

```text
##常用探索数据整体情况的方法
data.head()
data.shape
data.info()
data.columns
data.describe()
data['Class'].value_counts() 
data.isnull().sum()
data.isnull().sum().max() # 包含缺失最多的一列有多少
data.isnull().any()
data.isnull().values.any() # 数据中是否有缺失值
data.pivot_table(values='Amount', index='hour', columns='Class',aggfunc='count') ## 数据切分(透视表)
```

```text
## 以欺诈数据为例

#欺诈发生次数的直方分布图
Fraud_transacation = data[data["Class"]==1]
Normal_transacation= data[data["Class"]==0]
plt.figure(figsize=(10,6))
plt.subplot(121)
Fraud_transacation.Amount.plot.hist(title="Fraud Transacation")
plt.yscale('log')
plt.subplot(122)
Normal_transacation.Amount.plot.hist(title="Normal Transaction")
plt.yscale('log')


#欺诈次数与时间的相关性探索
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(Fraud.Time, Fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(Normal.Time, Normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

##数据特征之间的相关性
correlation_matrix = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True)
plt.show()

## 处理时间序列，找规律
data['hour'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)
data.pivot_table(values='Amount',index='hour',columns='Class',aggfunc='count')
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html

bins = np.arange(data['hour'].min(),data['hour'].max()+2)
# plt.figure(figsize=(15,4))
sns.distplot(data[data['Class']==0.0]['hour'],
             norm_hist=True,
             bins=bins,
             kde=False,
             color='b',
             hist_kws={'alpha':.5},
             label='Legit')
sns.distplot(data[data['Class']==1.0]['hour'],
             norm_hist=True,
             bins=bins,
             kde=False,
             color='r',
             label='Fraud',
             hist_kws={'alpha':.5})
plt.xticks(range(0,24))
plt.legend()
```

```text
### 比较省力的方法，通过包探测数据的分布情况
# https://github.com/pandas-profiling/pandas-profiling
import pandas_profiling

profile = df.profile_report(title="Credit Fraud Detector")
profile.to_file(output_file=Path("./credit_fraud_detector.html"))
```



> 欺诈数据的EDA Conclusion

> * The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases.
> * There is no missing value in the dataset.
> * The ‘Time’ and ‘Amount’ features are not transformed data.

#### Outliers Detection 异常值探索

* **Point anomalies**: A single instance of data is anomalous if it's too far off from the rest.
* **Contextual anomalies**: The abnormality is context specific. This type of anomaly is common in time-series data.

$$
IQR=Q_3​−Q_1 \\ Outliers:>Q_3​+k⋅IQR \\ Outliers:<Q_1​−k⋅IQR
$$

> The higher $$k$$ is \(eg: 3\), the less outliers will detect, and the lower $$k$$ is \(eg: 1.5\) the more outliers it will detect.

#### Unbalance  样本不平衡，解决方法：

* Collect more data
* Using the weights parameters 

```text
LogisticRegression(class_weight='balanced')
# How to choose weights?
```

* Changing the performance metric: 改变评价指标

准确率

1. [ ] 准确率：scikit-learn提供了accuracy\_score来计算：LogisticRegression.score\(\)  
2. [ ] 准确率是分类器预测正确性的比例，但是并不能分辨出假阳性错误和假阴性错误。例如：当正样本比例很低时，全部预测为负样本，准确率就很高，指标失去指示意义（1%为正样本，99%为负样本，全预测为负样本，准确率就有99%）

精确率和召回率

1. [ ] 精确率（precision）表示的是预测为正的样本中有多少是真正的正样本 

$$
Pre = \frac{TP}{TP+FP}
$$

1. [ ] 召回率 （Recall）表示的是样本中的正例有多少被预测正确了

$$
Recall = \frac{TP}{TP+FN}
$$

* [ ] 不同场景下，对于精确和召回的关注程度不同。医学检测，更关注召回；法律判决，更关注精确
* [ ] F1 是精确和召回的调和平均

$$
F1 = \frac{2*precision*recall}{precision+recall}
$$

* [ ] **对于类别不平衡问题，一般以数量较少的一类作为正类，这样准确和召回更具有指示意义\(即避免全部预测为占多数类别时指标就很高的情况出现\)**

**Area Under Curve \(AUC\)**--Area Under the curve of the Receiver Operating Characteristic \(AUROC\)

ROC 关注的两个指标

$$
TPR = \frac{TP}{TP+FN} = Recall Rate
$$

$$
FPR = \frac{FP}{TN+FP}
$$

直观上，TPR 代表能将正例分对的概率，FPR 代表将负例错分为正例的概率。在 ROC 空间中，每个点的横坐标是 FPR，纵坐标是 TPR，这也就描绘了分类器在 TP（真正率）和 FP（假正率）间的 trade-off

AUC（Area Under Curve）被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。含义为随机挑选一个正样本以及一个负样本，分类器判定正样本的值高于负样本的概率就是 AUC 值

1. AUC=1，完美分类器，采用这个预测模型时，不管设定什么阈值都能得出完美预测。绝大多数预测的场合，不存在完美分类器。
2. 0.5&lt;AUC&lt;1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
3. AUC=0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
4. AUC&lt;0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测，因此不存在 AUC&lt;0.5 的情况。

\*\*\*\*[**混淆矩阵**](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Confusion_matrix)\*\*\*\*

![](.gitbook/assets/image%20%284%29.png)

* [ ]  **Mean Absolute Error\(MAE\)**

$$
\text {MAE}=\frac{1}{N} \sum_{j=1}^{N}\left|y_{j}-\hat{y}_{j}\right|
$$

1. [ ]  **Mean Squared Error\(MSE\)**

$$
\text {MSE}=\frac{1}{N} \sum_{j=1}^{N}\left(y_{j}-\hat{y}_{j}\right)^{2}
$$

* [ ]  **Log Loss**

 AUC only takes into account **the order of probabilities** and hence it does not take into account the model’s capability to predict higher probability for samples more likely to be positive.

$$
\text {Log Loss}=-\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{i j} * \log \left(p(y_{i j})\right)
$$

> $$y_{ij}$$​: whether sample $$i$$belongs to class$$ j $$or not
>
> $$p(y_{i j})$$: the probability of sample$$ i $$belonging to class$$ j$$

若只有两类：则对数损失函数的公式简化为

$$
\text {Log Loss}=-\frac{1}{N} \sum_{i=1}^{N} y_{i} \cdot \log \left(p\left(y_{i}\right)\right)+\left(1-y_{i}\right) \cdot \log \left(1-p\left(y_{i}\right)\right)
$$

> 这时, $$y_i$$ 为输入实例 $$x_i $$的真实类别, $$p_i$$ 为预测输入实例 $$x_i$$ 属于类别 $$1$$ 的概率. 对所有样本的对数损失表示对每个样本的对数损失的平均值, 对于完美的分类器, 对数损失为$$ 0$$ .

* **Resampling the dataset 重抽样**
* [ ]  Under-Sampling:    从多数集中选出一部分数据与少数集重新组合成一个新的数据集

  1. [ ] **随机下采样：** 从多数类样本中随机选取一些剔除掉。这种方法的缺点是被剔除的样本可能包含着一些重要信息，致使学习出来的模型效果不好。
  2. [ ] **EasyEnsemble 和 BalanceCascade**采用集成学习机制来处理传统随机欠采样中的信息丢失问题**：**

     1. [ ]  **EasyEnsemble**将多数类样本随机划分成n个子集，每个子集的数量等于少数类样本的数量，这相当于欠采样。接着将每个子集与少数类样本结合起来分别训练一个模型，最后将n个模型集成，这样虽然每个子集的样本少于总体样本，但集成后总信息量并不减少
     2. [ ] BalanceCascade采用了有监督结合Boosting的方式（Boosting方法是一种用来提高弱分类算法准确度的方法,这种方法通过构造一个预测函数系列,然后以一定的方式将他们组合成一个预测函数）。在第n轮训练中，将从多数类样本中抽样得来的子集与少数类样本结合起来训练一个基学习器H，训练完后多数类中能被H正确分类的样本会被剔除。在接下来的第n+1轮中，从被剔除后的多数类样本中产生子集用于与少数类样本结合起来训练，最后将不同的基学习器集成起来。BalanceCascade的有监督表现在每一轮的基学习器起到了在多数类中选择样本的作用，而其Boosting特点则体现在每一轮丢弃被正确分类的样本，进而后续基学习器会更注重那些之前分类错误的样本。 

  3. [ ] **NearMiss**从多数类样本中选取最具代表性的样本用于训练，主要是为了缓解随机欠采样中的信息丢失问题。NearMiss采用一些启发式的规则来选择样本，根据规则的不同可分为3类：

     1. [ ]  **NearMiss-1**：选择到最近的$$K$$个少数类样本平均距离最近的多数类样本
     2. [ ]  **NearMiss-2**：选择到最远的$$K$$个少数类样本平均距离最近的多数类样本
     3. [ ]  **NearMiss-3**：对于每个少数类样本选择K个最近的多数类样本，目的是保证每个少数类样本都被多数类样本包围

* [ ]  Over-Sampling**过采样：**重复正比例数据，实际上没有为模型引入更多数据，过分强调正比例数据，会放大正比例噪音对模型的影响。一种简单的方式就是通过有放回抽样，缺点是如果特征少，会导致过拟合的问题。经过改进的过抽样方法通过在少数类中加入随机噪声、干扰数据或通过一定规则产生新的合成样本。
* [ ]  Under-Sampling Drawback**:**  Removing information that may be valuable. This could lead to underfitting and poor generalization to the test set.
* [ ]  **Under-sampling: Tomek links** are pairs of very close instances, but of opposite classes. Removing the instances of the majority class of each pair increases the space between the two classes, facilitating the classification process.

```text
## imblearn 处理数据不平衡的包
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

#print('Removed indexes:', id_tl)
plot_2d_space(X_tl, y_tl,X,y, 'Tomek links under-sampling')
```

* [ ]  **Over-sampling: SMOTE \(Synthetic Minority Oversampling Technique\)** consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.（从该少数类的全部 $$T $$个样本中找到样本$$ x_i$$ 的$$ k$$ 个近邻，然后从这$$ k$$ 个近邻中随机选择一个样本 $$x_i(nn)$$ ，再生成一个 0 到 1 之间的随机数 $$\epsilon_1$$ ，从而合成一个新样本 $$x_{i1}$$，重复$$N$$次）

#### 模型优化

![](.gitbook/assets/image%20%281%29.png)

