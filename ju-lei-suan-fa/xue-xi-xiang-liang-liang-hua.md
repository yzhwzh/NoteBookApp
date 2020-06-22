# 学习向量量化

1.  1. 与一般聚类算法不同，学习向量量化`Learning Vector Quantization:LVQ`假设数据样本带有类别标记，学习过程需要**利用样本的这些监督信息来辅助聚类**。
   2. 给定样本集 $$\mathbb D=\{(\mathbf{\vec x}_1,y_1),(\mathbf{\vec x}_2,y_2),\cdots,(\mathbf{\vec x}_N,y_N)\},\mathbf{\vec x}\in \mathcal X,y\in \mathcal Y$$ ，`LVQ`的目标是从特征空间中挑选一组样本作为原型向量 $$\{\mathbf{\vec p}_1,\mathbf{\vec p}_2,\cdots,\mathbf{\vec p}_Q\}$$  。
      * 每个原型向量代表一个聚类簇，簇标记 $$y_{p_q} \in \mathcal Y,q=1,2,\cdots,Q$$ 。即：簇标记从类别标记中选取。
      * 原型向量从特征空间中取得，它们不一定就是 $$\mathbb D$$ 中的某个样本。
   3. `LVQ`的想法是：通过从样本中挑选一组样本作为原型向量 $$\{\mathbf{\vec p}_1,\mathbf{\vec p}_2,\cdots,\mathbf{\vec p}_Q\}$$ ，可以实现对样本空间  $$\mathcal X$$ 的簇划分。
      * 对任意样本 $$\mathbf{\vec x}$$ ，它被划入与距离最近的原型向量所代表的簇中。
      * 对于每个原型向量 $$\mathbf{\vec p}_q$$ ，它定义了一个与之相关的一个区域 $$\mathbf R_q$$ ，该区域中每个样本与 $$\mathbf{\vec p}_q$$ 的距离都不大于它与其他原型向量 $$\mathbf{\vec p}_{q^{\prime}}$$ 的距离 $$\mathbf R_q=\{\mathbf{\vec x} \in \mathcal X \mid ||\mathbf{\vec x}-\mathbf{\vec p}_q||_2 \le \min_{q^{\prime} \ne q}||\mathbf{\vec x}-\mathbf{\vec p}_{q^{\prime}}||_2\}$$ 。
      * 区域 $$\{\mathbf R_1,\mathbf R_2,\cdots,\mathbf R_Q\}$$ 对样本空间  $$\mathcal X$$ 形成了一个簇划分，该划分通常称作 `Voronoi`剖分。
   4. 问题是如何从样本中挑选一组样本作为原型向量？ `LVQ`的思想是：
      * 首先挑选一组样本作为假设的原型向量。
      * 然后对于训练集中的每一个样本 $$\mathbf{\vec x}_i$$ ， 找出假设的原型向量中，距离该样本最近的原型向量 $$\mathbf{\vec p}_{q_i}$$ ： 
        * 如果 $$\mathbf{\vec x}_i$$ 的标记与 $$\mathbf{\vec p}_{q_i}$$ 的标记相同，则更新 $$\mathbf{\vec p}_{q_i}$$ ，将该原型向量更靠近 $$\mathbf{\vec x}_i$$ 。
        * 如果 $$\mathbf{\vec x}_i$$ 的标记与 $$\mathbf{\vec p}_{q_i}$$ 的标记不相同，则更新 $$\mathbf{\vec p}_{q_i}$$ ，将该原型向量更远离 $$\mathbf{\vec x}_i$$ 。
      * 不停进行这种更新，直到迭代停止条件（如以到达最大迭代次数，或者原型向量的更新幅度很小）。
   5. `LVQ`算法： 
      * 输入：
        * 样本集 $$\mathbb D=\{(\mathbf{\vec x}_1,y_1),(\mathbf{\vec x}_2,y_2),\cdots,(\mathbf{\vec x}_N,y_N)\}$$ 
        * 原型向量个数 $$Q$$ 
        * 各原型向量预设的类别标记 $$\{y_{p_1},y_{p_2},\cdots,y_{p_Q}\}$$ 
        * 学习率 $$\eta \in (0,1)$$ 
      * 输出：原型向量 $$\{\mathbf{\vec p}_1,\mathbf{\vec p}_2,\cdots,\mathbf{\vec p}_Q\}$$ 
      * 算法步骤：
        * 依次随机从类别 $$\{y_{p_1},y_{p_2},\cdots,y_{p_Q}\}$$ 中挑选一个样本，初始化一组原型向量 $$\{\mathbf{\vec p}_1,\mathbf{\vec p}_2,\cdots,\mathbf{\vec p}_Q\}$$  。
        * 重复迭代，直到算法收敛。迭代过程如下：
          * 从样本集 $$\mathbb D$$ 中随机选取样本 $$(\mathbf{\vec x}_i,y_i)$$ ，挑选出距离 $$(\mathbf{\vec x}_i,y_i)$$ 最近的原型向量 ： $$q_i=\arg\min_q ||\mathbf{\vec x}_i-\mathbf{\vec p}_q||$$ 。
          * 如果 $$\mathbf{\vec p_{q_i}}$$ 的类别等于 $$y_i$$ ，则： $$\mathbf{\vec p_{q_i}} \leftarrow \mathbf{\vec p_{q_i}}+\eta(\mathbf{\vec x}_i-\mathbf{\vec p_{q_i}})$$ 。
          * 如果 $$\mathbf{\vec p_{q_i}}$$ 的类别不等于 $$y_i$$ ，则： $$\mathbf{\vec p_{q_i}} \leftarrow \mathbf{\vec p_{q_i}}-\eta(\mathbf{\vec x}_i-\mathbf{\vec p_{q_i}})$$ 。
   6. 在原型向量的更新过程中：
      * 如果 $$\mathbf{\vec p_{q_i}}$$  的类别等于 $$y_i$$ ，则更新后， $$\mathbf{\vec p_{q_i}}$$与 $$\mathbf{\vec x}_i$$ 距离为： $$|| \mathbf{\vec p_{q_i}}-\mathbf{\vec x}_i||_2=||\mathbf{\vec p_{q_i}}+\eta(\mathbf{\vec x}_i-\mathbf{\vec p_{q_i}})- \mathbf{\vec x}_i||_2\     =(1-\eta)|| \mathbf{\vec p_{q_i}}-\mathbf{\vec x}_i||_2$$ 

        则更新后的原型向量 $$\mathbf{\vec p_{q_i}}$$ 距离  $$\mathbf{\vec x}_i$$ 更近。

      * 如果 $$\mathbf{\vec p_{q_i}}$$ 的类别不等于 $$y_i$$ ，则更新后， $$\mathbf{\vec p_{q_i}}$$与 $$\mathbf{\vec x}_i$$ 距离为： $$|| \mathbf{\vec p_{q_i}}-\mathbf{\vec x}_i||_2=||\mathbf{\vec p_{q_i}}-\eta(\mathbf{\vec x}_i-\mathbf{\vec p_{q_i}})- \mathbf{\vec x}_i||_2\     =(1+\eta)|| \mathbf{\vec p_{q_i}}-\mathbf{\vec x}_i||_2$$ 

        则更新后的原型向量 $$\mathbf{\vec p_{q_i}}$$ 距离  $$\mathbf{\vec x}_i$$ 更远。
   7. 这里有一个隐含假设：即计算得到的样本 $$\mathbf{\vec p_{q_i}} \pm\eta(\mathbf{\vec x}_i-\mathbf{\vec p_{q_i}})$$ （该样本可能不在样本集中） 的标记就是更新之前 $$\mathbf{\vec p_{q_i}}$$ 的标记。

      即：更新操作只改变原型向量的样本值，但是不改变该原型向量的标记。

