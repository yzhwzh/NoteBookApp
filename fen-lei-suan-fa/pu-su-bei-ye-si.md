# 朴素贝叶斯

### 相关基础知识:

贝叶斯定理:

$$
P(H|D) = \frac{P(H)P(D|H)}{P(D)} = \frac{P(H \cap D)}{P(D)}
$$

> \#$$P(H) $$称为先验概率，$$P(H|D)$$为后验概率，即需计算的概率；$$P(D|H)$$称为似然度，$$P(D)$$称为标准化常量

1）. 模型 朴素贝叶斯法通过训练数据集学习联合概率分布$$P(X,Y)$$. 

具体地，先验概率分布 $$P(Y=C_k)$$ $$k=1,2,...K$$_;_ 

条件概率分布 $$P(X=x|Y=C_k) = P(X^{(1)}=x^{(1)},...,X^{(N)}=x^{(N)}|Y=C_k)$$$$,k=1,2,...,K$$

假设$$x^{(j)}$$可取值有$$S_j$$个_，_$$j \in \{1,2,...,n\}$$ _,_$$Y$$可取值有$$K$$个，那么参数个数为$$K \prod_{j=1}^n S_j$$\_\_

**朴素贝叶斯对条件概率分布做了条件独立性的假设，由于这是一个较强的假设，朴素贝叶斯法由此得。条件独立性假设**  

$$
P(X=x | Y=C_k) = P(X^{(1)}=x^{(1)},...,X^{(N)}=x^{(N)}|Y=C_k) = \prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)
$$

**条件独立假设等于是说用于分类的特征在类确定的条件下都是条件独立的。这一假设使朴素贝叶斯法变得简单，但有时会牺牲一定的分类准确率** 

用于分类时，对给定的输入$$x$$,通过学习到的模型计算后验概率分布$$P(Y=C_k|X=x)$$,将后验概率最大的类作为$$x$$的类输出。后验概率计算根据贝叶斯定理进行

$$
P(Y=C_k | X=x) = \frac{P(X=x|Y=C_k)P(Y=C_k)}{\sum_k P(X=x|Y=C_k)P(Y=C_k)}
$$

故朴素贝叶斯公式：

$$
y=f(x)=argmax_{C_k} \frac{P(Y=C_k)\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=C_k)}{\sum_k P(Y=C_k)\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=C_k)}
$$

因为分母对所有$$C_k$$都是相同的_，_所以$$y=argmax_{C_k}P(Y=C_k)\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=C_k)$$**可以理解为分母是个常量？**

2\). 损失函数 属于分类问题，选择$$0-1$$损失函数

$$
L(Y,f(X))= \left\{\begin{array}{rl }+1,&Y \neq f(x)\\-1,&Y=f(x)\end{array} \right.
$$

 这时期望风险函数为：

$$
R_{exp}(f) = E[L(Y,f(x))] = E_x\sum_{k=1}^K[L(C_k,f(X))]P[C_k|X]
$$

为了使期望风险最小化，只需要对$$X=x$$逐个极小化，由此得到

$$
f(x) = argmin_{y\in Y} \sum_{k=1}^{K} L(C_k,f(X))P(C_k|X=x) \\= argmin_{y\in Y} \sum_{k=1}^{K} P(y \neq C_k | X=x) \\=argmin_{y\in Y} (1-P(y=C_k|X=x)) \\ = argmax_{y\in Y} P(y=C_k|X=x)
$$

即朴素贝叶斯法所采用的原理。

3、算法 极大似然估计 先验概率$$P(Y=C_k)$$ __的极大似然估计 

$$
P(Y=C_k) = \frac{\sum_{i=1}^N I(y_i = C_k)}{N},k=1,2,...,K
$$

设第$$j$$个特征$$x^{(j)}$$可能取值的集合为$${a_{j1},a_{j2},...,a_{jn}}$$，条件概率$$P(X^{(j)}=a_{jl}|Y=C_k)$$的极大似然估计是

$$
P(X^{(j)}=a_{jl}|Y=C_k) = \frac{\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=C_k)}{\sum_{i=1}^N I(y_i=C_k)}, j=1,2,...,n; l=1,2,...,S_j; k=1,2,...,K
$$

; $$x_i^{(j)}$$是第$$i$$个样本的第$$j$$个特征； $$a_{ij}$$是第$$j$$个特征可能取的第$$l$$个值，$$I$$为指示函数。

4、学习与分类算法

输入：训练数据

$$
T={(x_1,y_1),(x_2,y_2),...,(x_n,y_n)}
$$

,其中$$x_i = (x_i^1,x_i^2,...,x_i^n)^T$$,$$x_i^j$$是第$$i$$个样本的第$$j$$个特征_，_$$x_i^j \in {a_{j1},a_{j2},...,a_{jn}}, a_{jl} $$是第$$j$$个特征可能的第$$l$$个值，$$j=1,2,...,n, l=1,2,...,S_j, y_i \in {c_1,c_2,...,c_k}$$;实例 $$x$$； 

输出： 实例$$x$$的分类

步骤 :

1\) .  计算先验概率和条件概率

$$
P(Y=C_k) = \frac{\sum_{i=1}^N I(y_i = C_k)}{N},k=1,2,...,K
$$

$$
P(X^{(j)}=a_{jl}|Y=C_k) = \frac{\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=C_k)}{\sum_{i=1}^N I(y_i=C_k)}
$$

2\) . 对于给定的实例 $$x=(x^1,x^2,...,x^n)^T 计算$$

$$
P(Y=C_k)\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=C_k)
$$

3\) . 确定实例$$x$$的类$$y=argmax_{C_k}P(Y=C_k)\prod_{j=1}^n P（X^{(j)}=x^{(j)}|Y=C_k)$$

5、贝叶斯估计 即在朴素贝叶斯算法上加入拉普拉斯平滑 

$$
P(X^{(j)}=a_{jl}|Y=C_k) = \frac{\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=C_k)+\lambda}{\sum_{i=1}^N I(y_i=C_k)+S_j\lambda}
$$

 公式中$$\lambda>=0$$,等价于在随机变量各个取值的频数上赋予一个正数 $$\lambda>0$$。 当$$\lambda=0$$时，就是极大似然估计，常取$$\lambda = 1$$,这时称为拉普拉斯平滑

先验概率的贝叶斯估计 

$$
P(Y=C_k) = \frac{\sum_{i=1}^N I(y_i = C_k)+\lambda}{N+K\lambda}
$$

```text
##使用贝叶斯进行文档分类
## **朴素贝叶斯的两个假设** 1、特征之间相互独立；2、每个特征同等重要

from numpy import *

### 词集模型，每个词的出现与否作为一个特征，即只出现一次
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix,trainCategory):    
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)    # 计算p(y=1) 的概率
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()   ## 求得是y=1的条件下各个特征值的频率
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult   等式两边都取log，就从累积变为累加。
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def testingNB():
    listOPosts,listClasses =  loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

    
    
#### 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

### 使用朴素贝叶斯过滤垃圾邮件
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():             ## 留存交叉验证
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

#### 使用朴素贝叶斯分类器从个人广告中获取区域倾向

#一个RSS文件就是一段规范的XML数据，该文件一般以rss，xml或者rdf作为后缀。发布一个RSS文件（一般称为RSS Feed）后，这个RSS Feed中包含的信息就能直接被其他站点调用

def calcMostFreq(vocabList,fullText):
    ## 该函数遍历词汇表中的每个词并统计它在文本中的出现的此数，并从高到低进行排序，最后返回排序最高的30个单词
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]



```

