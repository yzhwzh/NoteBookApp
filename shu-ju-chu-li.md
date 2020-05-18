# 数据处理相关

#### 数据分析常规流程一般包括（A typical data science process might look like this）:

* Project Scoping / Data Collection  数据收集
* Exploratory Analysis  数据探索
* Data Cleaning  数据清洗 
  * 异常值检测
  * 数据不平衡
  * 评价指标
  * 采样
  * 验证集划分
  * 建模
* Feature Engineering  特征工程
* Model Training \(including cross-validation to tune hyper-parameters\)  模型训练，调参
* Project Delivery / Insights  结果解读

#### EDA：

* 对数据有一个基础认识（Gathering a basic sense of data）：
  * Shape\(Traing Size, Test size\)
  * Label\(Binary or Multi or Regression, Distribution\) 对标签数据进行探索，是二分类，多分类，还是回归预测，以及标签数据分布情况，是否平衡
  * Columns\(Meaning, Numerical or Time or Category\)
  * Null Values, how to deal with
  * Numerical variable: Distribution
  * Outliers

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
data.pivot_table(values='Amount', index='hour', columns='Class',aggfunc='count') ## 分组统计
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

#数据特征之间的相关性
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

