# EM 算法的推广

##  F 函数

1. `F`函数：假设隐变量 $$Z$$ 的概率分布为 $$\tilde P(  Z)$$ ，定义分布  $$\tilde P(  Z)$$ 与参数  $$\theta$$ 的函数 $$F(\tilde P,\theta)$$ 为： $$F(\tilde P,\theta)=\mathbb E_{\tilde P}[\log P(  Y, Z ;\theta)]+H(\tilde P)$$ 

   其中 $$H(\tilde P)=-\mathbb E_{\tilde P}\log \tilde P$$ 是分布 $$\tilde P(  Z)$$ 的熵。

   通常假定 $$P( Y,Z ;\theta)$$ 是 $$\theta$$ 的连续函数，因此 $$F(\tilde P,\theta)$$ 为 $$\tilde P(  Z)$$ 和 $$\theta$$ 的连续函数。

2. 函数 $$F(\tilde P,\theta)$$ 有下列重要性质：
   * 对固定的 $$\theta$$ ，存在唯一的分布 $$\tilde P_{\theta}(   Z )$$  使得极大化 $$F(\tilde P,\theta)$$ 。此时 $$\tilde P_{\theta}(   Z )=P(  Z \mid   Y;\theta)$$ ，并且 $$\tilde P_{\theta}$$ 随着 $$\theta$$ 连续变化。
   * 若 $$\tilde P_{\theta}(  Z )=P(   Z \mid  Y;\theta)$$ ， 则 $$F(\tilde P,\theta)=\log P( Y;\theta)$$ 。
3. 定理一：设 $$L(\theta)=\log P(\mathbb Y;\theta)$$ 为观测数据的对数似然函数， $$\theta^{<i>}$$ 为 `EM` 算法得到的参数估计序列，函数 $$F(\tilde P,\theta)=\sum_Y\mathbb E_{\tilde P}[\log P(Y,Z ;\theta)]+H(\tilde P)$$ ，则：
   * 如果 $$F(\tilde P,\theta)$$ 在 $$\tilde P^{*}(Z )$$ 和 $$\theta^{*}$$ 有局部极大值，那么 $$L(\theta)$$ 也在 $$\theta^{*}$$ 有局部极大值。
   * 如果 $$F(\tilde P,\theta)$$ 在 $$\tilde P^{*}(Z )$$ 和 $$\theta^{*}$$ 有全局极大值，那么 $$L(\theta)$$ 也在 $$\theta^{*}$$  有全局极大值。
4. 定理二：`EM`算法的一次迭代可由 `F` 函数的极大-极大算法实现：设  $$\theta^{<i>}$$ 为第 $$i$$ 次迭代参数 $$\theta$$ 的估计， $$\tilde P^{<i>}$$ 为第 $$i$$ 次迭代函数 $$\tilde P(Z )$$ 的估计。在第 $$i+1$$ 次迭代的两步为：
   * 对固定的 $$\theta^{<i>}$$ ，求 $$\tilde P^{<i+1>}$$ 使得 $$F(\tilde P,\theta^{<i>})$$ 极大化。
   * 对固定的 $$\tilde P^{<i+1>}$$ ，求 $$\theta^{<i+1>}$$ 使得 $$F(\tilde P^{<i+1>},\theta)$$ 极大化。

## GEM算法1

1. `GEM`算法1（`EM`算法的推广形式）：
   * 输入：
     * 观测数据 $$ \mathbb  Y=\{y_1,y_2,\cdots\}$$ 
     *  $$F$$ 函数
   * 输出：模型参数
   * 算法步骤：
     * 初始化参数 $$\theta^{<0>}$$ ，开始迭代。
     * 第 $$i+1$$ 次迭代：
       * 记  $$\theta^{<i>}$$ 为参数 $$\theta$$ 的估计值， $$\tilde P^{<i>}$$ 为函数 $$\tilde P$$ 的估计值。求 $$\tilde P^{<i+1>}$$ 使得 $$F(\tilde P,\theta^{<i>})$$ 极大化。
       * 求 $$\theta^{<i+1>}$$  使得 $$F(\tilde P^{<i+1>},\theta)$$ 极大化。
       * 重复上面两步直到收敛。
2. 该算法的问题是，有时候求 $$F(\tilde P^{<i+1>},\theta)$$ 极大化很困难。

## GEM算法2

1. `GEM`算法2（`EM`算法的推广形式）：
   * 输入：
     * 观测数据 $$ \mathbb  Y=\{y_1,y_2,\cdots\}$$ 
     *  $$Q$$ 函数
   * 输出：模型参数
   * 算法步骤：
     * 初始化参数  $$\theta^{<0>}$$ ，开始迭代。
     * 第 $$i+1$$  次迭代：
       * 记 $$\theta^{<i>}$$ 为参数 $$\theta$$ 的估计值， 计算 $$Q(\theta,\theta^{<i>})=\sum_{j=1}^N\left(\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log P(Y=y_j,Z;\theta) \right)$$ 
       * 求 $$\theta^{<i+1>}$$ 使得 $$Q(\theta^{<i+1>},\theta^{<i>}) \gt Q(\theta^{<i>},\theta^{<i>})$$ 
       * 重复上面两步，直到收敛。
2. 此算法不需要求 $$Q(\theta,\theta^{<i>})$$ 的极大值，只需要求解使它增加的 $$\theta^{<i+1>}$$ 即可。

## GEM算法3

1. `GEM`算法3（`EM`算法的推广形式）：
   * 输入：
     * 观测数据 $$ \mathbb  Y=\{y_1,y_2,\cdots\}$$ 
     *  函数
   * 输出：模型参数
   * 算法步骤：
     * 初始化参数  $$\theta^{<0>}=(\theta_1^{<0>},\theta_2^{<0>},\cdots,\theta_d^{<0>})$$ ，开始迭代
     * 第 $$i+1$$ 次迭代：
       * 记 $$\theta^{<i>}=(\theta_1^{<i>},\theta_2^{<i>},\cdots,\theta_d^{<i>})$$ 为参数 $$\theta=(\theta_1,\theta_2,\cdots,\theta_d)$$ 的估计值， 计算 $$Q(\theta,\theta^{<i>})=\sum_{j=1}^N\left(\sum_Z P(Z\mid Y=y_j;\theta^{<i>})\log P(Y=y_j,Z;\theta) \right)$$ 
       * 进行 $$d$$ 次条件极大化：
         * 首先在 $$\theta_2^{<i>},\cdots,\theta_d^{<i>}$$  保持不变的条件下求使得 $$Q(\theta,\theta^{<i>})$$ 达到极大的 $$\theta_1^{<i+1>}$$ 
         * 然后在 $$\theta_1=\theta_1^{<i+1>},\theta_j=\theta_j^{<i>},j=3,\cdots,d$$  的条件下求使得 $$Q(\theta,\theta^{<i>})$$ 达到极大的 $$\theta_2^{<i+1>}$$ 
         * 如此继续，经过 $$d$$ 次条件极大化，得到 $$\theta^{<i+1>}=(\theta_1^{<i+1>},\theta_2^{<i+1>},\cdots,\theta_d^{<i+1>})$$ ，使得 $$Q(\theta^{<i+1>},\theta^{<i>}) \gt Q(\theta^{<i>},\theta^{<i>})$$ 
       * 重复上面两步，直到收敛。
2. 该算法将 `EM` 算法的 `M` 步分解为  次条件极大化，每次只需要改变参数向量的一个分量，其余分量不改变。

