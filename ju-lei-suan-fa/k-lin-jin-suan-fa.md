---
description: K-Nearest Neighbor， 分类算法，不是K-means
---

# K邻近算法

**K 领近算法：给定一个训练集，对新的输入实例，在新的数据集中找到与该实例最领近的k个实例，这k个实例的多数属于某个类，就把该实例划分为这个类**

#### 算法 : 

1. 输入：训练数据集 $$T=\{ (x_1,y_1),(x_2,y_2),...,(x_n,y_n) \}$$ ，其中$$x_i\in \chi \subseteq R^n $$为实例的特征向量，$$y_i\in Y = \{c_1,c_2,c_3...c_k\}$$ 为实例的类别，$$i=1,2,...,N $$;实例特征向量$$x$$; 输出：实例$$x$$所属的类

1. 根据给定的距离度量，在训练集$$T$$中找出与$$x$$最领近的$$k$$个点，涵盖这$$k$$个点的$$x$$的邻域记作$$Nk(x)$$_;_ 
2. 在$$N_k(x)$$中根据分类决策规则（如多数表决）决定$$x$$的类别y: $$y=argmax{cj} \sum_{x\in N_k(x)} I(y_i=c_j),i=1,2,...,N;j=1,2,...,K$$ ;$$I$$为指示函数，即当$$y_i=c_i$$时$$I$$为$$1$$，否则$$I$$为0.

2. 距离度量

设特征空间$$\chi$$ 是$$n$$维实数向量空间$$R^n,x_i,x_j \in \chi, x_i=(x_i^{(1)},x_i^{(2)},...,x_i^{(n)})^T, x_j=(x_j^{(1)},x_j^{(2)},...,x_j^{(n)})^T, x_i,x_j$$_的_$$Lp$$_的距离为：_$$L_p(x_i,x_j) = (\sum{l=1}^n|x_i^{(l)}-x_j^{(l)}|^p)^{1/p}$$ 当$$p=2$$时，是欧式距离；当$$p=1$$时，是曼哈顿距离；当$$p=\infty$$，它是各个坐标距离的最大值。一般使用的是**欧式距离**。

3. $$k$$值的选择 

如果选择较小的$$k$$值，用较小的领域中的训练实例进行预测，学习的近似误差会减小，估计误差增大，预测结果会对近邻的实例点非常敏感，如果邻近的实例点恰巧是噪声，预测就会出错。换句话说，$$k$$值的减小就意味着整体模型变得复杂，容易发生过拟合。 如果选择较大的$$k$$值，误差会相反，但较远的训练实例也会对预测起作用，使预测发生错误。$$k$$值的增大就意味着整体的模型变得简单。 在应用中，$$k$$值一般选择一个比较小的数值，通常采用交叉验证法来选取最优的$$k$$值。

4. 分类决策规则

分类决策规则往往是多数表决，既由输入实例的$$k$$个邻近的训练实例中的多数类决定输入的类， 如果分类的损失函数为$$0-1$$；分类函数为$$ f:R^n->{c_1,c_2,...,c_k } $$_,那么误分类概率是_ $$P(Y\neq f(X))=1-P(Y = f(X))$$_;因而给定的实例_$$x \in \chi$$_,其最近邻的_$$k$$_个训练实例点构成集合_$$N_k(x)$$_,如果涵盖_$$N_k(x)$$_的区域类别是_$$c_j$$_,那么误分类率是_ $$1/k \sum{xi\in N_k(x)} I(y_i \neq c_j) = 1-1/k \sum{xi\in N_k(x)} I(y_i = c_j)$$_; 要使误分类率最小，就要使_$$1/k \sum{x_i\in N_k(x)} I(y_i = c_j)$$ 最大，所以**多数表决规则等价与经验风险最小化**。

5. 背景知识，树相关介绍 树的定义：一棵树是一些节点的集合，这个集合可以是空集；若非空，则一棵树由称做根的节点$$r$$以及$$0$$个或多个非空的树$$T_1,T_2,...,T_k$$组成，这些子树中每一棵的根都被来自根$$r$$的一条有向的边所连接。 树的实现：将每个节点的所有儿子都放到树节点的链表中。

常用的几个树结构： 

* 二叉树: 每个节点都不能多于两个儿子， 平均深度 $$O(\sqrt N)$$, 若是二叉查找树，其深度平均值是$$O(log N)$$。最坏的情况，输得深度能达到$$N-1$$ 

  * 用途： 

    * 表达式树，树叶为操作数，其他节点为操作符；遍历方法的不同，会导致最终树的赋值也不同，一般的方法为中序遍历。 
    * 查找树ADT-二叉查找树： 对于树中每个节点X，它的左子树中所有关键字值小于X的关键字值，右子树大于X的关键字值。 
    * AVL树，二叉查找树加上了平衡条件，每个节点的左子树与右子树的高度最多差$$1$$
    *  伸展树：当一个节点被访问后，他就要经过一系列AVL树的旋转被放到根上，保证对树的操作最多花费$$O（MlogN）$$

* $$B-树$$ ：树的根或者是一片树叶，或者其儿子数在$$2$$和$$M$$之间；除根外，所有非树叶节点的儿子数在$$[M/2]$$和$$M$$之间；所有树叶都在相同的深度上；$$M$$为深度。

6. $$K$$邻近，$$kd$$树，是二叉树，表示对$$k$$维空间的一个划分。构造$$kd$$树相当于不断地用垂直于坐标轴的超平面将$$k$$维空间切分，构成一系列的$$k$$维超矩形区域。 

输入：$$k$$维空间数据集$$T={ x_1,x_2,...,x_N }$$；其中$$x_i = {x_i^{(1)},x_i^{(2)},...,x_i^{(k)}},i=1,2,3...N$$ 

输出 $$kd$$树 

1. 构造根节点： 选择以$$x^{(1)}$$ 为坐标轴，以$$T$$中所有实例的$$x^{(1)}$$的中位数为切分点，将根节点对应的超矩形区域分为两个子区域.切分由通过切分点并与坐标轴$$x^{(1)}$$垂直的超平面实现。由根节点生成深度为$$1$$的左右子结点；左子结点对应坐标$$x^{(1)}$$小于切分点的区域，右子节点对应于大于切分点的区域。 
2. 重复：对深度为j的结点，选择$$x^{(l)}$$为切分的坐标轴，$$l=j(mod k)+1$$,以该结点区域中所有实例的$$x^{(l)}$$坐标的中位数为切分点，将该结点对应的超矩形区域分为两个子区域，切分由通过切分点并与坐标轴$$x^{(l)}$$垂直的超平面实现。 
3. 直到两个子区域没有实例存在时停止。从而形成$$kd$$树的区域划分。

7. 用$$kd$$树的最近邻搜索 输入：已构造的$$kd$$树；目标点$$x$$ 输出：$$x$$的最近邻

1. 在$$kd$$树中找出包含目标点$$x$$的叶节点：从根结点出发，递归地向下访问$$kd$$树。若目标点$$x$$当前维的坐标小于切分点的坐标，则移动到左子节点，都则移动到右子节点。直到子节点为叶节点为止。
2. 以此叶节点为“当前最近点”。
3. 递归地向上回退，在每个结点进行以下操作：

   1. 如果该结点保存的实例点比当前最近点距离目标点更近，则以该实例点为“当前最近点”
   2. 当前最近点一定存在于该结点一个子结点对应的区域。检查该子节点的父节点的另一子节点对应的区域是否有更近的点

4. 当回退到根节点时，搜索结果，最后的“当前最近点”即为x的最近邻点。

```text
##### KNN 算法

from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):  ## inX 输入向量，dataset 训练集，k选择邻近的数目
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  ##  numpy tile 函数，inX 列方向上重复一次，行重复dataSetSize次，即生成能减去所有实例点的一个矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)   
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  ## 返回数值从小到大的索引值   
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) ##items方法是可以将字典中的所有项，以列表方式返回。iteritems方法与items方法相比作用大致相同，只是它的返回值不是列表，而是一个迭代器,operator模块提供的itemgetter函数用于获取对象的哪些维的数据
    return sortedClassCount[0][0]
    
## 实例1 约会网站
##将人分为三类：不喜欢的人，魅力一般的人，极具魅力的人
##提供的3种特征值：每年获得飞行常客里程数；玩游戏视频所耗时间百分比；每周消费的冰淇淋公升数

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


#### 在处理这种不同取值范围的特征值时，我们通常采用的方法是将数值归一化， 即newvalue = (oldvalue-min)/(max-min)

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

## 实例2 手写识别系统

def img2vector(filename):                 ## 将filename里的数据变为一行，形成一个特征向量。
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


```

