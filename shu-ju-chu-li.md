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

