---
description: Feature Engineering Techniques
---

# 特征工程

**Feature Engineering : ceature engineering as creating new features from your existing ones to improve model performance.**

**Typical Enterprise Machine Learning Workflow**

![](../.gitbook/assets/image%20%2811%29.png)

> #### What is Not Feature Engineering? <a id="what-is-not-feature-engineering"></a>
>
> That means there are certain steps we do not consider to be feature engineering:
>
> * We do not consider **initial data collection** to be feature engineering.
> * Similarly, we do not consider **creating the target variable** to be feature engineering.
> * We do not consider removing duplicates, handling missing values, or fixing mislabeled classes to be feature engineering. We put these under **data cleaning**.
> * We do not consider **scaling or normalization** to be feature engineering because these steps belong inside the cross-validation loop.
> * Finally, we do not consider **feature selection or PCA** to be feature engineering. These steps also belong inside your cross-validation loop.

####  <a id="what-is-not-feature-engineering"></a>

#### 1. Indicator Variables <a id="indicator-variables"></a>

The first type of feature engineering involves using indicator variables to isolate key information.

* **Indicator variable from thresholds**: create an indicator variable for `age >= 18` to distinguish subjects who were the adult.
* **Indicator variable from multiple features**: You’re predicting real-estate prices and you have the features `n_bedrooms` and `n_bathrooms`. If houses with 2 beds and 2 baths command a premium as rental properties, you can create an indicator variable to flag them.
* **Indicator variable for special events**: You’re modeling weekly sales for an e-commerce site. You can create two indicator variables for the weeks of `11-11` and `6-18`.
* **Indicator variable for groups of classes**: You’re analyzing APP conversions and your dataset has the categorical feature traffic\_source. You could create an indicator variable for paid\_traffic by flagging observations with traffic source values of `HUAWEI` or `XIAOMI`.

#### 2. Interaction Features <a id="interaction-features"></a>

This type of feature engineering involves highlighting interactions between two or more features. Well, some features can be combined to provide more information than they would as individuals. Do you know `gender * father's height` and `gender * mather's height` in first homework?

**Note: We don’t recommend using an automated loop to create interactions for all your features. This leads to `feature explosion`.**

* **Sum of two features**
* **Difference between two features**
* **Product of two features**
* **Quotient of two features**

#### 3. Feature Representation <a id="feature-representation"></a>

Your data won’t always come in the ideal format. You should consider if you’d gain information by representing the same feature in a different way.

* **Date and time features**: Let’s say you have the feature `purchase_datetime`. It might be more useful to extract `purchase_day_of_week` and `purchase_hour_of_day`. You can also aggregate observations to create features such as `purchases_over_last_7_days`, `purchases_over_last_14_days`, `purchases_day_std`, `purchases_week_mean` etc.

```text
from datetime import date

data = pd.DataFrame({'date':
['01-01-2017',
'04-12-2008',
'23-06-1988',
'25-08-1999',
'20-02-1993',
]})

# Transform string to date
data['date'] = pd.to_datetime(data.date, format="%d-%m-%Y")

# Extracting Year
data['year'] = data['date'].dt.year

# Extracting Month
data['month'] = data['date'].dt.month

# Extracting passed years since the date
data['passed_years'] = date.today().year - data['date'].dt.year

# Extracting passed months since the date
data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month

# Extracting the weekday name of the date
data['day_name'] = data['date'].dt.day_name()

```

* **Binning \(分箱\)**: Binning can be applied on both categorical and numerical data. When there is a feature with many classes that have low sample counts. You can try grouping similar classes and then grouping the remaining ones into a single Other class \[**Grouping sparse classes**\].

![](../.gitbook/assets/image%20%2812%29.png)

The main motivation of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance. Every time you bin something, you sacrifice information and make your data more regularized.

```text
# Numerical Binning Example
data['bin'] = pd.cut(data['value'], bins=[0,30,70,100], labels=["Low", "Mid", "High"])

# Categorical Binning Example
conditions = [
    data['Country'].str.contains('Spain'),
    data['Country'].str.contains('Italy'),
    data['Country'].str.contains('Chile'),
    data['Country'].str.contains('Brazil')]

choices = ['Europe', 'Europe', 'South America', 'South America']

data['Continent'] = np.select(conditions, choices, default='Other')  # Grouping sparse classes
```

**For numerical columns, except for some obvious overfitting cases, binning might be redundant for some kind of algorithms, due to its effect on model performance. However, for categorical columns, the labels with low frequencies probably affect the robustness of statistical models negatively.**

* 特征二元化

  * 特征二元化的过程是将数值型的属性转换为布尔值的属性。通常用于假设属性取值为取值分布为伯努利分布的情形。
  * 特征二元化的算法比较简单。 对属性$$j$$指定一个阈值 $$\epsilon$$ 。
    * 如果样本在属性 $$j$$ 上的值大于等于$$\epsilon$$ ，则二元化之后为 1 。
    * 如果样本在属性 $$j$$ 上的值小于 $$\epsilon$$，则二元化之后为 0 。
  * 阈值$$\epsilon$$  是一个超参数，其选取需要结合模型和具体的任务来选择。

* **Labeled Encoding**: Interpret the categories as ordered integers \(mostly wrong\)
  * 对于非数值属性，如`性别：[男，女]、国籍：[中国，美国，英国]`等等，可以构建一个到整数的映射。如`性别：[男，女]`属性中，将`男`映射为整数 1、`女`映射为整数 0。

    该方法的优点是简单。但是问题是，在这种处理方式中无序的属性被看成有序的。`男`和`女`无法比较大小，但是`1`和`0`有大小。

    解决的办法是采用独热码编码`One-Hot Encoding`。
*  **One-Hot encoding**: Transform categories into individual binary \(0 or 1\) features ,  采用`N`位状态位来对`N`个可能的取值进行编码，每个取值都由独立的状态位来表示，并且在任意时刻只有其中的一位有效。

![](../.gitbook/assets/image%20%2810%29.png)



1. `One-Hot Encoding` 的优点： 
   * 能够处理非数值属性。
   * 在一定程度上也扩充了特征。如`性别`是一个属性，经过独热码编码之后变成了`是否男` 和 `是否女` 两个属性。
   * 编码后的属性是稀疏的，存在大量的零元分量。
2. 在决策树模型中，并不推荐对离散特征进行`one-hot`。 主要有两个原因：
   * 产生样本切分不平衡的问题，此时且分增益会非常小。

     如：`国籍`这个离散特征经过独热码编码之后，会产生`是否中国、是否美国、是否英国、...` 等一系列特征。在这一系列特征上，只有少量样本为`1`，大量样本为`0`。

     这种划分的增益非常小，因为拆分之后：

     * 较小的那个拆分样本集，它占总样本的比例太小。无论增益多大，乘以该比例之后几乎可以忽略。
     * 较大的那个拆分样本集，它几乎就是原始的样本集，增益几乎为零。

   * 影响决策树的学习。

     决策树依赖的是数据的统计信息。而独热码编码会把数据切分到零散的小空间上。在这些零散的小空间上，统计信息是不准确的，学习效果变差。

     本质是因为独热码编码之后的特征的表达能力较差的。该特征的预测能力被人为的拆分成多份，每一份与其他特征竞争最优划分点都失败。最终该特征得到的重要性会比实际值低。

*  **Frequency Encoding**: Encoding of categorical levels of feature to values between 0 and 1 based on their relative frequency.

![](../.gitbook/assets/image%20%287%29.png)

