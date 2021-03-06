# 支持向量机

1. 支持向量机\(`support vector machines` ：`SVM`）是一种二分类模型。它是定义在特征空间上的、间隔最大的线性分类器。**决定决策边界的数据叫做支持向量。**
   * 间隔最大使得支持向量机有别于感知机。

     **如果数据集是线性可分的，那么感知机获得的模型可能有很多个，而支持向量机选择的是间隔最大的那一个。**

   * 支持向量机还支持核技巧，从而使它成为实质上的非线性分类器。
2. 支持向量机支持处理线性可分数据集、非线性可分数据集。
   * 当训练数据线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机（也称作硬间隔支持向量机）。
   * 当训练数据近似线性可分时，通过软间隔最大化，学习一个线性分类器，即线性支持向量机（也称为软间隔支持向量机）。
   * 当训练数据不可分时，通过使用核技巧以及软间隔最大化，学习一个非线性分类器，即非线性支持向量机。
3. 当输入空间为欧氏空间或离散集合、特征空间为希尔伯特空间时，将输入向量从输入空间映射到特征空间，得到特征向量。

   支持向量机的学习是在特征空间进行的。

   * 线性可分支持向量机、线性支持向量机假设这两个空间的元素一一对应，并将输入空间中的输入映射为特征空间中的特征向量。
   * 非线性支持向量机利用一个从输入空间到特征空间的非线性映射将输入映射为特征向量。
     * 特征向量之间的内积就是核函数，使用核函数可以学习非线性支持向量机。
     * 非线性支持向量机等价于隐式的在高维的特征空间中学习线性支持向量机，这种方法称作核技巧。

4. 欧氏空间是有限维度的，希尔伯特空间为无穷维度的。
   * 欧式空间 $$\subseteq$$ 希尔伯特空间 $$\subseteq$$ 内积空间 $$\subseteq$$ 赋范空间。
     * 欧式空间，具有很多美好的性质。
     * 若不局限于有限维度，就来到了希尔伯特空间。

       从有限到无限是一个质变，很多美好的性质消失了，一些非常有悖常识的现象会出现。

     * 如果再进一步去掉完备性，就来到了内积空间。
     * 如果再进一步去掉"角度"的概念，就来到了赋范空间。此时还有“长度”和“距离”的概念。
   * 越抽象的空间具有的性质越少，在这样的空间中能得到的结论就越少
   * 如果发现了赋范空间中的某些性质，那么前面那些空间也都具有这个性质。

