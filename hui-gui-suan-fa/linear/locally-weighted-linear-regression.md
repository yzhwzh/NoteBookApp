---
description: Locally Weighted Linear Regression
---

# 局部加权线性回归

### 介绍：

线性回归有可能出现欠拟合，因为它求得是具有最小均方误差的无偏估计。有些方法允许在估计中引入一些偏差，从而降低预测的均方误差。其中一个方法即是局部加权线性回归\(LWLR,Locally Weighted Linear Regression\)，通过对待预测点附近的每个点赋予一定的权重，在这个子集上基于最小均方差进行普通回归。很多情况下，加权矩阵即核，选择的是高斯核：

$$
w_{x_j}^{(i)} = e^{-\frac{(x^{(i)}-x_j)^2}{2\sigma^2}}
$$

表示意义为，离 $$x_j$$ 越近的点，权重越高，越远的点，权重越低。

### 损失函数：

$$
J_{(x_j)}(\theta) = \frac{1}{2n}\sum_{i=1}^nw_{x_j}^{(i)}(y_i-\hat y_i)^2
$$

通过矩阵可以表示为\(W为一个对角阵\)：

$$
J_{x_j}(\theta) = (Y-X\beta_{x_j})^TW_{x_j}(Y-X\beta_{x_j})
$$

同理求偏导，令之为0：

$$
\begin{aligned}
\epsilon^TW_{x_j}X = 0 \\ (Y-X\beta)^TW_{x_j}X = 0\\ X^TW_{x_j}^T(Y-X\beta) = X^TW_{x_j}Y- X^TW_{x_j}X\beta = 0\\\hat\beta_{x_j} = (X^TW_{x_j}X)^{-1}X^TW_{x_j}Y
\end{aligned}
$$

进而预测：

$$
\hat y_j = x_j*\hat\beta_{x_j}
$$

> 每一行都会计算一个权重矩阵，对角线上的值表示数据其他行与该行的距离权重；因而每一个预测值都会有各自的系数矩阵

#### 局部线性回归代码：

```text
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    m = shape(xMat).[0]
    yMat = mat(yArr).T
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linag.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I*(xMat^T*weights*yMat)
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat
```



