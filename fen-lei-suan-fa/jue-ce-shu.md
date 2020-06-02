---
description: Decision Tree
---

# 决策树

#### 相关基础知识 

* **信息熵**：

        熵的概念首先在热力学中引入，用于表述热力学第二定律。波尔兹曼研究得到，热力学熵与微观状态数目的对数之间存在联系，并给出了公式： $$S=K \log W$$ 。信息熵的定义与上述这个热力学的熵，虽然不是一个东西，但是有一定的联系。熵在信息论中代表随机变量不确定度的度量。一个离散型随机变量 X 的熵 $$H(X) $$定义为：

$$
H(X) = -\sum_{i=1}^n p_i \log p_i; P(X=x_i) = p_i
$$

；若$$p_i=0$$,则$$0\log 0 = 0$$; **以2为底数，熵的单位为比特\(bit\)，以e为底数，熵的单位称作纳特\(nat\),熵只依赖于**$$X$$**的分布，与**$$X$$**的取值无关，故也可记作**$$H(P)$$。 直觉上，信息量等于传输该信息所用的代价，这个也是通信中考虑最多的问题。 信息论之父克劳德·香农，总结出了信息熵的三条性质： 

1. 单调性，即发生概率越高的事件，其所携带的信息熵越低。极端案例就是“太阳从东方升起”，因为为确定事件，所以不携带任何信息量。从信息论的角度，认为这句话没有消除任何不确定性。 
2. 非负性，即信息熵不能为负。这个很好理解，因为负的信息，即你得知了某个信息后，却增加了不确定性是不合逻辑的。 
3. 累加性，即多随机事件同时发生存在的总不确定性的量度是可以表示为各事件不确定性的量度的和。写成公式就是：事件 $$X=A, Y=B$$同时发生，两个事件相互独立 $$P(X=A,Y=B)=P(X=A)P(Y=B)$$_，那么信息熵_ $$H(A,B) = H(A)+H(B)$$。

> 补充一下，如果两个事件不相互独立，那么满足: $$H(A,B) = H(A)+H(B)-I(A,B)$$；$$I(A,B)$$为互信息，代表一个随机变量包含另一个随机变量信息量的度量

* **条件熵**：

        设有随即变量 $$ (X,Y)$$，其联合概率分布为 

$$
P(X=x_i,Y=y_i) = P_{ij},i=1,2,...,n;j=1,2,...,m
$$

；条件熵$$H（Y|X）$$表示在已知随机变量$$X$$的条件下，随机变量$$Y$$的不确定性， 定义为$$H(Y|X)=\sum_{i=1}^np_iH(Y|X=x_i)$$，这里$$p_i=P(X=x_i),i=1,2,...,n$$。当熵和条件熵中的概率由数据估计（特别是极大似然估计）得到时，所对应的熵与条件熵分别称为经验熵和经验条件熵。此时如果有0概率，令$$0log0 = 0$$。

* **信息增益**：

        特征$$A$$对训练数据集$$D$$的信息增益$$g（D, A）$$，定义为集合$$D$$的经验熵$$H（D）$$与特征$$A$$给定条件下$$D$$的经验条件熵$$H（D|A）$$之差，即 $$Ig（D, A）= H(D)-H(D|A)$$; 一般也称为互信息。**表示由于特征A而使得对数据集D的分类的不确定性减少的程度**，信息增益大的特征具有更强的分类能力。 根据信息增益准则的特征选择方法是：对训练数据集$$D$$，计算其每个特征的信息增益，并比较他们的大小，选择信息增益最大的特征。 

        设训练数据集为$$D$$，$$|D|$$表示其样本个数。设有$$K$$个类 $$C_k,k=1,2,...,K;|C_k|$$为属于类$$C_k$$的样本个数，$$\sum_{k=1}^K |C_k| = |D| $$。设特征$$A$$有$$n$$个不同的取值$${a_1,a_2,...,a_n}$$， 根据特征$$A$$的取值将$$D$$划分为$$n$$个子集$$D_1,D_2,...,D_n, |D_i|$$为$$D_i$$的样本个数，$$\sum_{i=1}^n |Di| = |D|$$. 记子集$$D_i$$中属于类$$C_k$$的样本的集合为$$D_{ik}$$,即$$D_{ik} = D_i \cap C_k$$

* **信息增益算法**： 

输入：训练数据集和特征$$A$$ 输出：特征$$A$$对训练数据集$$D$$的信息增益$$g(D,A)$$ ，步骤：

* 计算数据集$$D$$的经验熵, $$H(D)=-\sum_{k=1}^K \frac{|C_k|}{D} \log_2 \frac{|C_k|}{D}$$ 
* 计算特征A对数据集的经验条件熵，

$$
H（D|A）= \sum_{i=1}^n \frac{|D_i|}{|D|}H(Di)= -\sum_{i=1}^n \frac{|D_i|}{|D|}\sum_{i=1}^K \frac{D_{ik}}{D_i}\log_2\frac{D_{ik}}{D_i}
$$

> 在这里$$H(D|A=a_i)$$的信息熵可以理解为$$H(D_i)$$

* 计算信息增益$$ g(D,A) = H(D)-H(D|A)$$

信息增益比： $$g_R(D，A) = \frac{g(D,A)}{H_A(D)}$$；其中$$ H_A(D) = -\sum{i=1}^n\frac{|D_i|}{|D|} \log_2 \frac{|D_i|}{|D|}$$

#### 分类决策树模型

        是一种描述对实例进行分类的树形结构。决策树由结点\(node\)和有向边\(directed edge\)组成。结点有两种类型：内部节点和叶节点。内部节点表示一个特征或属性，叶节点表示一个类。 **决策树学习本质上是从训练数据集中归纳出一组分类规则。从另一个角度看，决策树是有训练数据集估计条件概率模型** 。决策树学习的损失函数通常是**正则化的极大似然函数**，决策树学习的策略是以损失函数为目标函数的最小化。 因为从所有可能的决策树中选取最优决策树是NP完全问题，所以现实中决策树学习算法通常采用启发式方法，近似求解这一；最优问题，这样得到的决策树是此最优的（NP完全问题:多项式复杂程度的非确定性问题）

#### 决策树学习算法

        包含 **特征选择、决策树的生成与决策树的剪枝过程；决策树的生成对应于模型的局部选择 （有可能发生过拟合现象），决策树的剪枝对应于模型的全局选择**； 决策树学习常用的算法有**ID3, C4.5与CART** ；

**1\). ID3 算法**

ID3算法的核心是在决策树各个节点上应用信息增益准则选择特征，递归地构建决策树。

输入：训练数据集D，特征集A，阈值$$\varepsilon$$

输出：决策树T 

1. 若$$D$$中所有实例属于同一类$$C_k$$，则$$T$$为单结点树，并将类$$C_k$$作为该结点的类标记，返回T； 
2. 若$$A \neq \varnothing$$，则T为单节点树，并将$$D$$中实例数最大的类$$C_k$$作为该结点的类标记，返回T； 
3. 否则，计算$$A$$中各特征对$$D$$的信息增益，选择信息增益最大的特征$$A_g$$ 
4. 如果$$A_g$$的信息增益小于阈值$$\varepsilon$$，则置$$T$$为单节点树，并将$$D$$中实例数最大的类$$C_k$$作为该结点的类标记，返回T 。
5. 对第$$i$$个子节点，以$$D_i$$为训练集，以$$A-{A_g}$$为特征集，递归的调用1~5，得到子树$$T_i$$，返回$$T_i$$。

* C4.5的生成算法： C4.5算法与ID3算法相似，只是用的信息增益比

#### 决策树的剪枝：

        往往通过极小化决策树整体的损失函数来实现。设树的叶节点个数为$$|T|$$，$$t$$是树$$T$$的叶节点，叶节点有$$N_t$$_个样本点，其中_$$k$$_类的样本点有_$$N_{tk}$$个，$$H_t(T)$$为叶节点$$t$$上的经验熵_，_$$\alpha >= 0$$；决策树的学习损失函数可以定义为$$C{\alpha}(T)=\sum{t=1}^{|T|}N_tH_t(T)+\alpha |T|$$; 其中经验熵$$H_t(T) = -\sum_k \frac{N_{tk}}{N_t}\log_2 \frac{N_{tk}}{N_t}$$；

        令$$C（T）=\sum{t=1}^{|T|}N_tH_t(T)$$，则$$C{\alpha}(T)=C（T）+\alpha |T|$$；

> $$C(T)$$表示模型对训练数据的预测误差，即模型与训练数据的拟合程度，$$|T|$$表示模型复杂度，参数$$\alpha >= 0 $$控制两者之间的影响，较大的$$\alpha$$促使选择教简单的模型，较小的$$\alpha$$促使选择较复杂的模型，$$\alpha=0$$ 意味着只考虑与训练数据的拟合程度，不考虑模型的复杂度。

**决策树剪枝算法**： 

输入：生成算法产生的整个树T，参数$$\alpha$$ 

输出：修剪后的子树$$T{\alpha}$$ __

1. 计算每个节点的经验熵 
2. 递归地从树的叶节点向上回缩； 设一组叶节点回缩到其父节点之前与之后的整体树分别为$$T_B$$与$$T_A$$,其对应的损失函数值分别是$$C{\alpha}(TB) $$与 $$C{\alpha}(TA)$$,如果$$C{\alpha}(TA)<= C{\alpha}(TB)$$，则进行剪枝，父节点变为新的叶节点 
3. 返回2，直至不能继续为止，得到损失函数最小的子树$$T{\alpha}$$

**2\). CART算法** 

        分类与回归树模型由Breiman等人在1984年提出，是应用广泛的决策树学习方法。CART同样由特征选择、树的生成及剪枝组成。CART 是在给定输入随机变量X条件下输出随机变量Y的条件概率分布的学习方法。

        CART假设决策树是二叉树，左分支是‘是’；右分支是‘否’。决策树生成：基于训练数据集生成决策树，生成的决策树要尽量大。决策树剪枝：用验证数据集对已生成的树进行剪枝并选择最优子树，用损失函数最小作为剪枝标准CART 生成：回归树用平方误差最小化准则，分类树用基尼指数最小化准则。

最小二乘回归树生成算法 

输入：训练数据集D； 

输出：回归树f\(x\);

在训练数据集合所在的输入空间中，递归地将每个区域划分为两个子区域并决定每个子区域上的输出值，构建二叉决策树； 

1\). 选择最优切分变量$$j$$与切分点$$s$$，求解

$$
min_{j,s}[min_{c_1}\sum_{x_i\in {R_1(j,s)}}(y_i-c_1)^2+min_{c_2}\sum_{x_i\in {R_2(j,s)}}(y_i-c_2)^2]
$$

遍历变量$$j$$，对固定的切分变量$$j$$扫描切分点$$s$$，选择使式达到最小值的对$$（j,s）$$ 

2\). 用选定的对$$(j,s)$$划分区域并决定相应的输出值：

$$
R_1(j,s)=\{x|x^{j}<=s\},R_2(j,s)=\{x|x^{j}>s\};\hat{c}_m = \frac{1}{N_m}\sum \limits_{x_i\in{R_m}(j,s)}y_i,x\in{R_m},m=1,2
$$

3\).  继续对两个子区域调用步骤1\)，2\)，直至满足停止条件 

4\).  将输入空间划分为$$M$$个区域$${R1,R_2,...,R_M}$$_生成决策树：_$$f(x) = \sum\limits{m=1}^M\hat{c_m}I(x\in R_m)$$\_\_

**分类树的生成** 

        分类问题中，假设有$$K$$个类，样本点属于第$$K$$类的概率为$$p_k$$_,_则概率分布的基尼指数定义为:

$$
Gini(p) = \sum\limits{k=1}^Kpk(1-p_k)=1-\sum\limits{k=1}^Kpk^2
$$

_；_对于二分类问题，若样本点属于第1个类的概率是$$p$$，则概率分布的基尼指数为$$Gini(p) = 2p(1-p)$$；对于给定的样本集合$$D$$，其基尼指数为$$Gini(D) = 1-\sum{k=1}^K(\frac{|C_k|}{|D|})^2$$,这里$$C_k$$是$$D$$中属于第$$k$$类的样本子集，$$K$$是类的个数；在特征$$A$$的条件下，集合$$D$$的基尼指数定义为:

$$
Gini(D,A) = \frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D2)
$$

基尼指数$$Gini(D)$$表示集合$$D$$的不确定性，基尼指数$$Gini(D,A)$$表示经$$A=a$$分割后集合$$D$$的不确定性，一般选择基尼指数小的特征值进行分类。

CART剪枝 大同小异

```text
#### 以下是采用的ID3算法，直接用的信息熵
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):    ###每一行的最后一个字符作为键值，计算概率
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting 将选中的标签删掉作为分割符号
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]   #stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel
#pickle模块中的两个主要函数是dump()和load()：
#dump()函数接受一个文件句柄和一个数据对象作为参数，把数据对象以特定的格式保存到给定的文件中。当我们使用load()函数从文件中取出已保存的对象时，pickle知道如何恢复这些对象到它们本来的格式。
#dumps()函数执行和dump() 函数相同的序列化。取代接受流对象并将序列化后的数据保存到磁盘文件，这个函数简单的返回序列化的数据。
#loads()函数执行和load() 函数一样的反序列化。取代接受一个流对象并去文件读取序列化后的数据，它接受包含序列化后的数据的str对象, 直接返回的对象。
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

###画决策树树形
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

#def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

#createPlot(thisTree)
```

