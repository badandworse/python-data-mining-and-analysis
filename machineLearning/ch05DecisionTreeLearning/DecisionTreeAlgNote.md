# 决策树模型与学习

1. 决策树学习的损失函数通常是正则化的极大似然函数。它的策略是以损失函数为目标函数的最小化。

2. **决策树定义**:节点和有向边组成。节点有两种：内部节点与叶节点。内部节点表示一个特征或属性，叶节点表示一个类。

## 特征选择

1. 特征选择的准则是信息增益或信息增益比

2. **熵**的定义:表示随机变量不确定性的度量。设*X*是一个取有限个值的离散随机变量，其概率分布为:

    $P(X=x_i)=p_id$

    随机变量*X*的熵为:

    $H(X)=-\sum_{i=1}^{n} p_i log_2p_i$

    其中$0log_20=0$

    熵越大，随机变量的不确定性越大。
3. 熵和条件熵的概率有数据估计得到时，所对应的熵与条件熵分别成为经验熵和经验条件熵:

    $H(Y|X)=\sum_{i=1}^{n}p_iH(Y|X=x_i)$

### 信息增益

1. **信息增益**表示得知特征*X*的信息而使得类*Y*的信息的不确定性减少的程度：

    $g(D,A)=H(D)-H(D|A)$

    当*g(D,A)*越大，即特征*A*的信息增益越大时，*H(D|A)*越小，特征*A*对数据集*D*分类效果越明显。

### 信息增益比

**定义**:特征*A*对训练数据集*D*的信息增益比$g_r(D,A)$定义为其信息增益*g(D,A)*与训练数据集*D*关于特征*A*的值的熵$H_A(D)$只比:

$g_R(D,A)=g(D,A)/H_A(D)$

$H_A(D)=-\sum_{i=1}^{n}|D_i|/|D|log_2|D_i|/|D|$,n 是特征A取值的个数

## 常用的决策树分类算法

常用的决策树分类算法有ID3、C4.5和CART

1. ID3:

    输入:训练数据集*D*,特征集*A*,阀值$\beta$:

    (1)若*D*中所有实例属性同一类$C_k$，则为单节点树，将类$C_k$作为该节点的类标记，返回*T*

    (2)若*A*为空，则*T*为单节点树，并将*D*中实例数最大的类$C_k$作为该结点的类标记，返回*T*。

    (3)如果非单节点树，计算所有*A*对*D*的信息增益，选择信息增益大的特征$A_g$。

    (4)若最大的值小于阀值$\beta$，则是单节点树，选取同(2)

    (5)对于$A_g$的每一可能值$a_i$,将集合*D*划分为若干个非空子集$D_i$,将$D_i$中实例数最大的子集作为标记，构建子节点。重复这一过程，直到无法再划分。

2. C4.5:

    与ID3类似，不过是用信息增益比来选择特征。

3. CART算法:

    既可以用于回归也可以用于分类。回归是使用平方误差最小化准则，对分类树用基尼指数最小化准则，进行特征筛选。

    **基尼指数**:分类问题中，假设有*K*个类，样本点属于第*K*类的概率为$p_k$，则概率分布的基尼系数为:

    $Gini(p)=\sum_{i=1}^{K}p_k(1-p_k)=1-\sum_{i=1}^{K}p_k^2$

    对于二类分类(是或不是，即0，1)问题，概率分布的基尼系数:

    $Gini(p)=2p(1-p)$

    对于给定的样本集合，其基尼系数为

    $Gini(p)=1-\sum_k=1^K(|C_k|/|D|)^2$

    $C_K$是*D*中属于第*k*类的样本子集，*K*是类的个数

    如果样本集合*D*根据特征*A*是否取某一可能值被分为$D_1$和$D_2$两部分，则在特征*A*的条件下，集合*D*的基尼指数定义为:

    $Gini(D,A)=|D_1|/|D|Gini(D_1)+|D_2|/|D|Gini(D_2)$

    *CART算法步骤*:
    与ID3类似，不过标准是基尼指数最小的特征及其对应的切分点作为最优特征与最优切分点。
