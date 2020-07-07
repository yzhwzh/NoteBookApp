# EM算法原理

## 观测变量和隐变量

1. 令 $$Y$$ 表示观测随机变量， $$\mathbb Y=\{y_1,y_2,\cdots,y_N\}$$ 表示对应的数据序列；令 $$Z$$ 表示隐随机变量，  $$\mathbb Z=\{z_1,z_2,\cdots,z_N\}$$ 表示对应的数据序列。

    $$Y$$ 和 $$Z$$ 连在一起称作完全数据，观测数据 $$Y$$ 又称作不完全数据。

2. 假设给定观测随机变量 $$Y$$ ，其概率分布为 $$P(Y;\theta)$$ ，其中 $$\theta$$ 是需要估计的模型参数，则不完全数据 $$Y$$ 的似然函数是 $$P(\mathbb Y;\theta)$$ ， 对数似然函数为 $$L(\theta)=\log P(\mathbb Y;\theta)$$ 。

   假定 $$Y$$ 和 $$Z$$ 的联合概率分布是 $$P(Y,Z;\theta)$$ ，完全数据的对数似然函数是 $$\log P(\mathbb Y,\mathbb Z;\theta)$$ ，则根据每次观测之间相互独立，有： $$\log P(\mathbb Y;\theta)=\sum_i \log P(Y=y_i;\theta)\\ \log P(\mathbb Y,\mathbb Z;\theta)=\sum_i \log P(Y=y_i,Z=z_i;\theta)$$ 

3. 由于 $$Y$$ 发生，根据最大似然估计，则需要求解对数似然函数： $$L(\theta)=\log P(\mathbb Y;\theta)=\sum_{i=1}\log P(Y=y_i;\theta) =\sum_{i=1}\log\sum_Z P(Y=y_i,  Z;\theta)\\ =\sum_{i=1}\log\left[\sum_Z P(Y=y_i \mid Z;\theta)P(Z;\theta)\right]$$ 

   的极大值。其中 $$\sum_Z P(Y=y_i,Z;\theta)$$ 表示对所有可能的 $$Z$$ 求和，因为边缘分布 $$P(Y)=\sum_Z P(Y,Z)$$ 。

   该问题的困难在于：该目标函数包含了未观测数据的的分布的积分和对数。

## EM算法

**原理**

1. `EM` 算法通过迭代逐步近似极大化  $$L(\theta)$$ 。

   假设在第  $$i$$ 次迭代后， $$\theta$$ 的估计值为： $$\theta^{<i>}$$ 。则希望 $$\theta$$  新的估计值能够使得 $$L(\theta)$$ 增加，即： $$L(\theta) \gt L(\theta^{<i>})$$  。

   为此考虑两者的差： $$L(\theta) - L(\theta^{<i>})=\log P(\mathbb Y;\theta)-\log P(\mathbb Y;\theta^{<i>})$$ 。

   > 这里 $$\theta^{<i>}$$ 已知，所以 $$\log P(\mathbb Y;\theta^{<i>})$$  可以直接计算得出。

2. `Jensen` 不等式：如果 $$f$$ 是凸函数， $$x$$ 为随机变量，则有： $$\mathbb E[f(x)]\le f(\mathbb E[x])$$ 。
   * 如果  $$f$$ 是严格凸函数，当且仅当 $$x$$  是常量时，等号成立。

     ![jensen](http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/imgs/EM/jensen.jpg)

   * 当 $$\lambda_i$$ 满足 $$ \lambda_j \ge 0,\sum_j \lambda_j=1$$ 时，将 $$\lambda_j$$ 视作概率分布。

     设随机变量 $$y$$ 满足概率分布  $$p(y=y_j)=\lambda_j$$ ，则有： $$\log\sum_j\lambda_j y_j \ge \sum_j\lambda_j\log y_j$$ 。
3. 考虑到条件概率的性质，则有 $$\sum_Z  P(Z \mid Y;\theta)=1$$ 。因此有： $$L(\theta) - L(\theta^{<i>})=\sum_{j}\log\sum_Z P(Y=y_j,  Z;\theta) - \sum_{j}\log  P(Y=y_j; \theta^{<i>})\\ =\sum_{j}\left[\log\sum_ZP(Z\mid Y=y_j;\theta^{<i>})\frac{P(Y=y_j,  Z;\theta)}{P(Z\mid Y=y_j;\theta^{<i>})} - \log  P(Y=y_j;\theta^{<i>}) \right]\\ \ge\sum_{j}\left[\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log\frac{P(Y=y_j,  Z;\theta)}{P(Z\mid Y=y_j;\theta^{<i>})} - \log P(Y=y_j;\theta^{<i>}) \right]\\ =\sum_{j}\left[\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log\frac{P(Y=y_j\mid Z;\theta)P(Z;\theta)}{P(Z\mid Y=y_j;\theta^{<i>})} - \log P(Y=y_i;\theta^{<i>})\times 1 \right]\\ =\sum_{j}\left[\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log\frac{P(Y=y_j\mid Z;\theta)P(Z;\theta)}{P(Z\mid Y=y_j;\theta^{<i>})} \\ - \log P(Y=y_j;\theta^{<i>})\times \sum_Z P(Z\mid Y=y_j;\theta^{<i>}) \right]\\ =\sum_{j}\left[\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log\frac{P(Y=y_j\mid Z;\theta)P(Z;\theta)}{P(Z\mid Y=y_j;\theta^{<i>}) P(Y=y_j;\theta^{<i>})} \right]$$ 

   等号成立时，需要满足条件： $$P(Z\mid Y=y_j;\theta^{<i>})=\frac 1 {n_Z}\\ \frac{P(Y=y_j,  Z;\theta)}{P(Z\mid Y=y_j;\theta^{<i>})}=\text{const}$$ 

   其中 $$n_Z$$ 为随机变量 $$Z$$ 的取值个数。

4. 令 ：

![](../.gitbook/assets/image%20%2829%29.png)

则有： $$L(\theta) \ge B(\theta,\theta^{<i>}) $$ ，因此 $$B(\theta,\theta^{<i>}) $$ 是 $$L(\theta^{<i>})$$ 的一个下界。

* 根据定义有： $$L(\theta^{<i>})=B(\theta^{<i>},\theta^{<i>})$$  。因为此时有： $$\frac{P( Y=y_j\mid  Z ;\theta^{<i>})P(  Z ;\theta^{<i>})}{P( Z \mid  Y=y_j;\theta^{<i>})P(Y=y_j;\theta^{<i>})}=\frac{P( Y=y_j, Z ;\theta^{<i>})}{P(Y=y_j,Z ;\theta^{<i>})} =1$$ 
* 任何可以使得 $$B(\theta,\theta^{<i>}) $$ 增大的 $$\theta$$ ，也可以使 $$L(\theta)$$ 增大。

  为了使得 $$L(\theta)$$ 尽可能增大，则选择使得 $$B(\theta,\theta^{<i>}) $$ 取极大值的  $$\theta$$ ： $$\theta^{<i+1>}=\arg\max_\theta B(\theta,\theta^{<i>})$$ 。

5. 求极大值：

![](../.gitbook/assets/image%20%2828%29.png)

其中： $$L(\theta^{<i>}),P(Z\mid Y=y_j;\theta^{<i>}) P(Y=y_j;\theta^{<i>})$$  与  $$\theta$$ 无关，因此省略。

**算法**

1. `EM` 算法：
   * 输入：

     * 观测变量数据 $$ \mathbb  Y=\{y_1,y_2,\cdots,y_N\}$$ 
     * 联合分布 $$P(Y,Z ;\theta)$$ ，以及条件分布 $$P( Z \mid  Y;\theta)$$ 

     > 联合分布和条件分布的形式已知（比如说高斯分布等），但是参数未知（比如均值、方差）

   * 输出：模型参数 $$\theta$$ 
   * 算法步骤：
     * 选择参数的初值 $$\theta^{<0> }$$ ，开始迭代。
     * `E`步：记 $$\theta^{<i>}$$ 为第 $$i$$ 次迭代参数 $$\theta$$ 的估计值，在第 $$i+1$$ 步迭代的 `E` 步，计算： $$Q(\theta,\theta^{<i>})=\sum_{j=1}^N \mathbb E_{P(Z\mid Y=y_j;\theta^{<i>})}\log P(Y=y_j,Z ;\theta)\\ =\sum_{j=1}^N\left(\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log P(Y=y_j,Z;\theta) \right)$$ 

       其中 $$\mathbb E_{P(Z\mid Y=y_j;\theta^{<i>})}\log P(Y=y_j,Z ;\theta)$$ 表示：对于观测点 $$Y=y_j$$ ， $$\log P(Y=y_j,Z ;\theta)$$  关于后验概率 $$P(Z\mid Y=y_j;\theta^{<i>})$$ 的期望。

     * `M`步：求使得 $$Q(\theta,\theta^{<i>})$$ 最大化的 $$\theta$$ ，确定 $$i+1$$ 次迭代的参数的估计值 $$\theta^{<i+1>}$$ $$\theta^{<i+1>}=\arg\max_\theta Q(\theta,\theta^{<i>})$$ 
     * 重复上面两步，直到收敛。
2. 通常收敛的条件是：给定较小的正数 $$\varepsilon_1,\varepsilon_2$$ ，满足： $$||\theta^{<i+1>}-\theta^{<i>}|| \lt \varepsilon_1$$ 或者  $$||Q(\theta^{<i+1>},\theta^{<i>})-Q(\theta^{<i>},\theta^{<i>})|| \lt \varepsilon_2$$ 。
3. $$Q(\theta,\theta^{<i>})$$ 是算法的核心，称作 $$Q$$ 函数。其中：
   * 第一个符号表示要极大化的参数（未知量）。
   * 第二个符号表示参数的当前估计值（已知量）。
4. `EM`算法的直观理解：`EM`算法的目标是最大化对数似然函数 $$L(\theta)=\log P(\mathbb Y)$$ 。
   * 直接求解这个目标是有问题的。因为要求解该目标，首先要得到未观测数据的分布  $$P(Z \mid  Y;\theta)$$ 。如：身高抽样问题中，已知身高，需要知道该身高对应的是男生还是女生。

     但是未观测数据的分布就是待求目标参数 $$\theta$$  的解的函数。这是一个“鸡生蛋-蛋生鸡” 的问题。

   * `EM`算法试图多次猜测这个未观测数据的分布  。

     每一轮迭代都猜测一个参数值 ，该参数值都对应着一个未观测数据的分布 $$P(Z \mid Y;\theta)$$ 。如：已知身高分布的条件下，男生/女生的分布。

   * 然后通**过最大化某个变量来求解参数值。这个变量就是**  $$B(\theta,\theta^{<i>})$$ **变量，它是真实的似然函数的下界 。** 
     * 如果猜测正确，则 $$B$$ 就是真实的似然函数。
     * 如果猜测不正确，则 $$B$$ 就是真实似然函数的一个下界。
5. 隐变量估计问题也可以通过梯度下降法等算法求解，但由于求和的项数随着隐变量的数目以指数级上升，因此代价太大。
   * `EM`算法可以视作一个非梯度优化算法。
   * 无论是梯度下降法，还是`EM` 算法，都容易陷入局部极小值。

**收敛性定理**

1. 定理一：设  $$P(\mathbb Y;\theta)$$ 为观测数据的似然函数， $$\theta^{<i>}$$ 为`EM`算法得到的参数估计序列， $$P(\mathbb Y;\theta^{<i>})$$ 为对应的似然函数序列，其中  $$i=1,2,\cdots$$ 。

   则： $$P(\mathbb Y;\theta^{<i>})$$ 是单调递增的，即： $$P(\mathbb Y;\theta^{<i+1>}) \ge P(\mathbb Y;\theta^{<i>})$$ 。

2. 定理二：设 $$L(\theta)=\log P(\mathbb Y;\theta)$$ 为观测数据的对数似然函数， $$\theta^{<i>}$$ 为`EM`算法得到的参数估计序列， $$L(\theta^{<i>})$$ 为对应的对数似然函数序列，其中 $$i=1,2,\cdots$$ 。
   * 如果 $$P(\mathbb Y;\theta)$$ 有上界，则 $$L(\theta^{<i>})$$ 会收敛到某一个值 $$L^{*}$$ 。
   * 在函数 $$Q(\theta,\theta^{<i>})$$ 与 $$L(\theta)$$ 满足一定条件下，由 `EM` 算法得到的参数估计序列 $$\theta^{<i>}$$ 的收敛值 $$\theta^{*}$$ 是 $$L(\theta)$$ 的稳定点。

     > 关于“满足一定条件”：大多数条件下其实都是满足的。
3. 定理二只能保证参数估计序列收敛到对数似然函数序列的稳定点 $$L^{*}$$ ，不能保证收敛到极大值点。
4. `EM`算法的收敛性包含两重意义：

   * 关于对数似然函数序列 $$L(\theta^{<i>})$$ 的收敛。
   * 关于参数估计序列 $$\theta^{<i>}$$ 的收敛。

   前者并不蕴含后者。

5. 实际应用中，`EM` 算法的参数的初值选择非常重要。
   * 参数的初始值可以任意选择，但是 `EM` 算法对初值是敏感的，选择不同的初始值可能得到不同的参数估计值。
   * 常用的办法是从几个不同的初值中进行迭代，然后对得到的各个估计值加以比较，从中选择最好的（对数似然函数最大的那个）。
6. `EM` 算法可以保证收敛到一个稳定点，不能保证得到全局最优点。其优点在于：简单性、普适性。



