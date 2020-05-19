# 类别不平衡问题

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

![](../.gitbook/assets/image%20%288%29.png)

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

* [ ] **再缩放**

1. 假设对样本 $$\vec x$$  进行分类时，预测为正类的概率为 $$p$$ 。常规的做法是将 $$p$$ 与一个阈值，比如 0.5 ， 进行比较。 如果 $$p>0.5$$ 时，就判别该样本为正类。

   概率 $$p$$ 刻画了样本为正类的可能性， 几率 $$\frac{p}{1-p}$$ 刻画了正类可能性与反类可能性的比值。

2. 当存在类别不平衡时，假设 $$N^+$$  表示正类样本数目， $$N^-$$ 表示反类样本数目，则观测几率是 $$\frac{N^+}{N^-}$$ 。

   假设训练集是真实样本总体的无偏采样，因此可以用观测几率代替真实几率。于是只要分类器的预测几率高于观测几率就应该判断为正类。即如果 $$\frac{p}{1-p} > \frac{N^+}{N^-}$$ ， 则预测为正类。

3. 通常分类器都是基于概率值来进行预测的，因此需要对其预测值进行调整。在进行预测的时候，令： $$\frac{\tilde{p}}{1-\tilde{p}}=\frac{p}{1-p} \times \frac{N^+}{N^-}$$ 然后再将 $$\tilde{p}$$  跟阈值比较。这就是类别不平衡学习的一个基本策略：再缩放`rescalling` 。
4. 再缩放虽然简单，但是由于“训练集是真实样本总体的无偏采样”这个假设往往不成立，所以无法基于训练集观测几率来推断出真实几率。



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

* [ ]  **Over-sampling: SMOTE \(Synthetic Minority Oversampling Technique\)** consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.（从该少数类的全部 $$T $$个样本中找到样本$$ x_i$$ 的$$ k$$ 个近邻，然后从这$$ k$$ 个近邻中随机选择一个样本 $$x_i(nn)$$ ，再生成一个 0 到 1 之间的随机数 $$\epsilon_1$$ ，与该样本相加，从而合成一个新样本 $$x_{i1}$$，重复$$N$$次）
* \*\*\*\*

