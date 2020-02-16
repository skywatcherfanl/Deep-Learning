
Task03：过拟合、欠拟合及其解决方案；梯度消失、梯度爆炸；循环神经网络进阶
=================================================================

# 1 过拟合、欠拟合及其解决方案
## 1.1 训练误差和泛化误差
 
在机器学习中，我们需要区分训练误差（training error）和泛化误差（generalization error）。通俗来讲，前者指模型在训练数据集上表现出的误差，后者指模型在任意一个测试数据样本上表现出的误差的期望，并常常通过测试数据集上的误差来近似。计算训练误差和泛化误差可以使用之前介绍过的损失函数，例如线性回归用到的平方损失函数和softmax回归用到的交叉熵损失函数。    
在机器学习里，我们通常假设训练数据集和测试数据集里的每一个样本都是从同一个概率分布中相互独立地生成的。基于该独立同分布假设，给定任意一个机器学习模型（含参数），它的训练误差的期望和泛化误差都是一样的。一般情况下，由训练数据集学到的模型参数会使模型在训练数据集上的表现优于或等于在测试数据集上的表现。由于无法从训练误差估计泛化误差，一味地降低训练误差并不意味着泛化误差一定会降低。机器学习模型应关注降低泛化误差。

## 1.2 模型选择

在机器学习中，通常需要评估若干候选模型的表现并从中选择模型。这一过程称为模型选择（model selection）。可供选择的候选模型可以是有着不同超参数的同类模型。以多层感知机为例，我们可以选择隐藏层的个数，以及每个隐藏层中隐藏单元个数和激活函数。为了得到有效的模型，我们通常要在模型选择上下一番功夫。下面，我们来描述模型选择中经常使用的验证数据集（validation data set）。

### 1.2.1 验证数据集

从严格意义上讲，测试集只能在所有超参数和模型参数选定后使用一次。不可以使用测试数据选择模型，如调参。由于无法从训练误差估计泛化误差，因此也不应只依赖训练数据选择模型。鉴于此，我们可以预留一部分在训练数据集和测试数据集以外的数据来进行模型选择。这部分数据被称为验证数据集，简称验证集（validation set）。例如，我们可以从给定的训练集中随机选取一小部分作为验证集，而将剩余部分作为真正的训练集。

然而在实际应用中，由于数据不容易获取，测试数据极少只使用一次就丢弃。因此，实践中验证数据集和测试数据集的界限可能比较模糊。从严格意义上讲，除非明确说明，否则本书中实验所使用的测试集应为验证集，实验报告的测试结果（如测试准确率）应为验证结果（如验证准确率）。

### 1.2.2 $K$折交叉验证

由于验证数据集不参与模型训练，当训练数据不够用时，预留大量的验证数据显得太奢侈。一种改善的方法是$K$折交叉验证（$K$-fold cross-validation）。在$K$折交叉验证中，我们把原始训练数据集分割成$K$个不重合的子数据集，然后我们做$K$次模型训练和验证。每一次，我们使用一个子数据集验证模型，并使用其他$K-1$个子数据集来训练模型。在这$K$次训练和验证中，每次用来验证模型的子数据集都不同。最后，我们对这$K$次训练误差和验证误差分别求平均。

## 1.3 欠拟合和过拟合

接下来，我们将探究模型训练中经常出现的两类典型问题：一类是模型无法得到较低的训练误差，我们将这一现象称作欠拟合（underfitting）；另一类是模型的训练误差远小于它在测试数据集上的误差，我们称该现象为过拟合（overfitting）。在实践中，我们要尽可能同时应对欠拟合和过拟合。虽然有很多因素可能导致这两种拟合问题，在这里我们重点讨论两个因素：模型复杂度和训练数据集大小。

### 1.3.1 模型复杂度

为了解释模型复杂度，我们以多项式函数拟合为例。给定一个由标量数据特征$x$和对应的标量标签$y$组成的训练数据集，多项式函数拟合的目标是找一个$K$阶多项式函数

$$
\hat{y} = b + \sum_{k=1}^K x^k w_k
$$

来近似 $y$。在上式中，$w_k$是模型的权重参数，$b$是偏差参数。与线性回归相同，多项式函数拟合也使用平方损失函数。特别地，一阶多项式函数拟合又叫线性函数拟合。

因为高阶多项式函数模型参数更多，模型函数的选择空间更大，所以高阶多项式函数比低阶多项式函数的复杂度更高。因此，高阶多项式函数比低阶多项式函数更容易在相同的训练数据集上得到更低的训练误差。给定训练数据集，模型复杂度和误差之间的关系通常如图1.1所示。给定训练数据集，如果模型的复杂度过低，很容易出现欠拟合；如果模型复杂度过高，很容易出现过拟合。应对欠拟合和过拟合的一个办法是针对数据集选择合适复杂度的模型。

<div align=center>
<img width="350" src="image/task03/3.11_capacity_vs_error.svg"/>
</div>
<div align=center>图1.1 模型复杂度对欠拟合和过拟合的影响</div>


### 1.3.2 训练数据集大小

影响欠拟合和过拟合的另一个重要因素是训练数据集的大小。一般来说，如果训练数据集中样本数过少，特别是比模型参数数量（按元素计）更少时，过拟合更容易发生。此外，泛化误差不会随训练数据集里样本数量增加而增大。因此，在计算资源允许的范围之内，我们通常希望训练数据集大一些，特别是在模型复杂度较高时，例如层数较多的深度学习模型。

## 小结

* 由于无法从训练误差估计泛化误差，一味地降低训练误差并不意味着泛化误差一定会降低。机器学习模型应关注降低泛化误差。
* 可以使用验证数据集来进行模型选择。
* 欠拟合指模型无法得到较低的训练误差，过拟合指模型的训练误差远小于它在测试数据集上的误差。
* 应选择复杂度合适的模型并避免使用过少的训练样本。

-----------
> [原书传送门](https://zh.d2l.ai/chapter_deep-learning-basics/underfit-overfit.html)

# 2 梯度消失、梯度爆炸

深度模型有关数值稳定性的典型问题是消失（vanishing）和爆炸（explosion）。

## 2.1 当神经网络的层数较多时，模型的数值稳定性容易变差 

假设一个层数为$L$的多层感知机的第$l$层$\boldsymbol{H}^{(l)}$的权重参数为$\boldsymbol{W}^{(l)}$，输出层$\boldsymbol{H}^{(L)}$的权重参数为$\boldsymbol{W}^{(L)}$。为了便于讨论，不考虑偏差参数，且设所有隐藏层的激活函数为恒等映射（identity mapping）$\phi(x) = x$。给定输入$\boldsymbol{X}$，多层感知机的第$l$层的输出$\boldsymbol{H}^{(l)} = \boldsymbol{X} \boldsymbol{W}^{(1)} \boldsymbol{W}^{(2)} \ldots \boldsymbol{W}^{(l)}$。此时，如果层数$l$较大，$\boldsymbol{H}^{(l)}$的计算可能会出现衰减或爆炸。举个例子，假设输入和所有层的权重参数都是标量，如权重参数为0.2和5，多层感知机的第30层输出为输入$\boldsymbol{X}$分别与$0.2^{30} \approx 1 \times 10^{-21}$（消失）和$5^{30} \approx 9 \times 10^{20}$（爆炸）的乘积。当层数较多时，梯度的计算也容易出现消失或爆炸。

## 2.2 随机初始化模型参数

在神经网络中，通常需要随机初始化模型参数。下面我们来解释这样做的原因。

回顾多层感知机一节描述的多层感知机。为了方便解释，假设输出层只保留一个输出单元$o_1$（删去$o_2$和$o_3$以及指向它们的箭头），且隐藏层使用相同的激活函数。如果将每个隐藏单元的参数都初始化为相等的值，那么在正向传播时每个隐藏单元将根据相同的输入计算出相同的值，并传递至输出层。在反向传播中，每个隐藏单元的参数梯度值相等。因此，这些参数在使用基于梯度的优化算法迭代后值依然相等。之后的迭代也是如此。在这种情况下，无论隐藏单元有多少，隐藏层本质上只有1个隐藏单元在发挥作用。因此，正如在前面的实验中所做的那样，我们通常将神经网络的模型参数，特别是权重参数，进行随机初始化。

<div align=center>
<img width="350" src="image/task03/2.1.svg"/>
</div>
<div align=center>图2.1 参数初始化</div>


###  2.2.1 PyTorch的默认随机初始化

随机初始化模型参数的方法有很多。在线性回归的简洁实现中，我们使用`torch.nn.init.normal_()`使模型`net`的权重参数采用正态分布的随机初始化方式。不过，PyTorch中`nn.Module`的模块参数都采取了较为合理的初始化策略（不同类型的layer具体采样的哪一种初始化方法的可参考[源代码](https://github.com/pytorch/pytorch/tree/master/torch/nn/modules)），因此一般不用我们考虑。


### 2.2.2 Xavier随机初始化

还有一种比较常用的随机初始化方法叫作Xavier随机初始化。
假设某全连接层的输入个数为$a$，输出个数为$b$，Xavier随机初始化将使该层中权重参数的每个元素都随机采样于均匀分布

$$
U\left(-\sqrt{\frac{6}{a+b}}, \sqrt{\frac{6}{a+b}}\right).
$$

它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。

## 2.3 考虑环境因素

### 2.3.1 协变量偏移

这里我们假设，虽然输入的分布可能随时间而改变，但是标记函数，即条件分布P（y∣x）不会改变。虽然这个问题容易理解，但在实践中也容易忽视。数据使用的是猫和狗的真实的照片，但是在测试时，我们被要求对猫和狗的卡通图片进行分类。

|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![Image Name](https://cdn.kesci.com/upload/image/q5jg8j72fl.jpg?imageView2/0/w/200/h/200)|![Image Name](https://cdn.kesci.com/upload/image/q5jg993za3.jpg?imageView2/0/w/200/h/200)|![Image Name](https://cdn.kesci.com/upload/image/q5jg9tqs4s.jpg?imageView2/0/w/200/h/200)|![Image Name](https://cdn.kesci.com/upload/image/q5jga6mnsk.jpg?imageView2/0/w/200/h/200)|

测试数据：

|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![Image Name](https://cdn.kesci.com/upload/image/q5jgat5lsd.png?imageView2/0/w/200/h/200)|![Image Name](https://cdn.kesci.com/upload/image/q5jgbaoij8.png?imageView2/0/w/200/h/200)|![Image Name](https://cdn.kesci.com/upload/image/q5jgbswvbb.png?imageView2/0/w/200/h/200)|![Image Name](https://cdn.kesci.com/upload/image/q5jgc5j7zv.png?imageView2/0/w/200/h/200)
|

显然，这不太可能奏效。训练集由照片组成，而测试集只包含卡通。在一个看起来与测试集有着本质不同的数据集上进行训练，而不考虑如何适应新的情况，这是不是一个好主意。不幸的是，这是一个非常常见的陷阱。

统计学家称这种协变量变化是因为问题的根源在于特征分布的变化（即协变量的变化）。数学上，我们可以说P（x）改变了，但P（y∣x）保持不变。尽管它的有用性并不局限于此，当我们认为x导致y时，协变量移位通常是正确的假设。


### 2.3.2 标签偏移

#### 标签偏移可以简单理解为测试时出现了训练时没有的标签。    
当我们认为导致偏移的是标签P（y）上的边缘分布的变化，但类条件分布是不变的P（x∣y）时，就会出现相反的问题。当我们认为y导致x时，标签偏移是一个合理的假设。例如，通常我们希望根据其表现来预测诊断结果。在这种情况下，我们认为诊断引起的表现，即疾病引起的症状。有时标签偏移和协变量移位假设可以同时成立。例如，当真正的标签函数是确定的和不变的，那么协变量偏移将始终保持，包括如果标签偏移也保持。有趣的是，当我们期望标签偏移和协变量偏移保持时，使用来自标签偏移假设的方法通常是有利的。这是因为这些方法倾向于操作看起来像标签的对象，这（在深度学习中）与处理看起来像输入的对象（在深度学习中）相比相对容易一些。

病因（要预测的诊断结果）导致 症状（观察到的结果）。  

训练数据集，数据很少只包含流感p(y)的样本。  

而测试数据集有流感p(y)和流感q(y)，其中不变的是流感症状p(x|y)。


### 2.3.3 概念偏移

另一个相关的问题出现在概念转换中，即标签本身的定义发生变化的情况。这听起来很奇怪，毕竟猫就是猫。的确，猫的定义可能不会改变，但我们能不能对软饮料也这么说呢？事实证明，如果我们周游美国，按地理位置转移数据来源，我们会发现，即使是如图所示的这个简单术语的定义也会发生相当大的概念转变。

<div align=center>
<img width="350" src="image/task03/ay327-mdifp.svg"/>
</div>
<div align=center>图2.3 美国软饮料名称的概念转变 </div>

如果我们要建立一个机器翻译系统，分布P（y∣x）可能因我们的位置而异。这个问题很难发现。另一个可取之处是P（y∣x）通常只是逐渐变化。


# 3 循环神经网络进阶
## 3.1 门控循环神经⽹络GRU（Gated Recurrent Neural Network）
* RNN存在的问题：梯度较容易出现衰减或爆炸（BPTT）
* ⻔控循环神经⽹络：捕捉时间序列中时间步距离较⼤的依赖关系

当时间步数较大或者时间步较小时，循环神经网络的梯度较容易出现衰减或爆炸。虽然裁剪梯度可以应对梯度爆炸，但无法解决梯度衰减的问题。通常由于这个原因，循环神经网络在实际中较难捕捉时间序列中时间步距离较大的依赖关系。

门控循环神经网络（gated recurrent neural network）的提出，正是为了更好地捕捉时间序列中时间步距离较大的依赖关系。它通过可以学习的门来控制信息的流动。其中，门控循环单元（gated recurrent unit，GRU）是一种常用的门控循环神经网络。

下面将介绍门控循环单元的设计。它引入了重置门（reset gate）和更新门（update gate）的概念，从而修改了循环神经网络中隐藏状态的计算方式。

### 3.1.1 重置门和更新门

如图3.1所示，门控循环单元中的重置门和更新门的输入均为当前时间步输入$\boldsymbol{X}_ t$与上一时间步隐藏状态$\boldsymbol{H}_ {t-1}$，输出由激活函数为sigmoid函数的全连接层计算得到。

<div align=center>
<img width="500" src="image/task03/6.7_gru_1.svg"/>
</div>
<div align=center>图3.1 门控循环单元中重置门和更新门的计算</div>


具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$\boldsymbol{X}_ t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$\boldsymbol{H}_ {t-1} \in \mathbb{R}^{n \times h}$。重置门$\boldsymbol{R}_ t \in \mathbb{R}^{n \times h}$和更新门$\boldsymbol{Z}_ t \in \mathbb{R}^{n \times h}$的计算如下：

$$
\begin{aligned}
\boldsymbol{R}_ t = \sigma(\boldsymbol{X}_ t \boldsymbol{W}_ {xr} + \boldsymbol{H}_ {t-1} \boldsymbol{W}_ {hr} + \boldsymbol{b}_ r), 
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol{Z}_ t = \sigma(\boldsymbol{X}_ t \boldsymbol{W}_ {xz} + \boldsymbol{H}_ {t-1} \boldsymbol{W}_ {hz} + \boldsymbol{b}_ z),
\end{aligned}
$$

其中$\boldsymbol{W}_ {xr}, \boldsymbol{W}_ {xz} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_ {hr}, \boldsymbol{W}_ {hz} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_ r, \boldsymbol{b}_ z \in \mathbb{R}^{1 \times h}$是偏差参数。sigmoid函数可以将元素的值变换到0和1之间。因此，重置门$\boldsymbol{R}_ t$和更新门$\boldsymbol{Z}_ t$中每个元素的值域都是$[0, 1]$。

* 重置门有助于捕捉时间序列里短期的依赖关系
* 更新门有助于捕捉时间序列里长期的依赖关系

### 3.1.2 候选隐藏状态

接下来，门控循环单元将计算候选隐藏状态来辅助稍后的隐藏状态计算。如图3.2所示，我们将当前时间步重置门的输出与上一时间步隐藏状态做按元素乘法（符号为$\odot$）。如果重置门中元素值接近0，那么意味着重置对应隐藏状态元素为0，即丢弃上一时间步的隐藏状态。如果元素值接近1，那么表示保留上一时间步的隐藏状态。然后，将按元素乘法的结果与当前时间步的输入连结，再通过含激活函数tanh的全连接层计算出候选隐藏状态，其所有元素的值域为$[-1, 1]$。

<div align=center>
<img width="500" src="image/task03/6.7_gru_2.svg"/>
</div>
<div align=center>图3.2 门控循环单元中候选隐藏状态的计算</div>

具体来说，时间步$t$的候选隐藏状态$\tilde{\boldsymbol{H}}_ t \in \mathbb{R}^{n \times h}$的计算为

$$\tilde{\boldsymbol{H}}_ t = \text{tanh}(\boldsymbol{X}_ t \boldsymbol{W}_ {xh} + \left(\boldsymbol{R}_ t \odot \boldsymbol{H}_ {t-1}\right) \boldsymbol{W}_ {hh} + \boldsymbol{b}_ h),$$

其中$\boldsymbol{W}_ {xh} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_ {hh} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_ h \in \mathbb{R}^{1 \times h}$是偏差参数。从上面这个公式可以看出，重置门控制了上一时间步的隐藏状态如何流入当前时间步的候选隐藏状态。而上一时间步的隐藏状态可能包含了时间序列截至上一时间步的全部历史信息。因此，重置门可以用来丢弃与预测无关的历史信息。

### 3.1.3 隐藏状态

最后，时间步$t$的隐藏状态$\boldsymbol{H}_ t \in \mathbb{R}^{n \times h}$的计算使用当前时间步的更新门$\boldsymbol{Z}_ t$来对上一时间步的隐藏状态$\boldsymbol{H}_ {t-1}$和当前时间步的候选隐藏状态$\tilde{\boldsymbol{H}}_ t$做组合：

$$\boldsymbol{H}_ t = \boldsymbol{Z}_ t \odot \boldsymbol{H}_ {t-1}  + (1 - \boldsymbol{Z}_ t) \odot \tilde{\boldsymbol{H}}_ t.$$

<div align=center>
<img width="500" src="image/task03/6.7_gru_3.svg"/>
</div>
<div align=center>图3.3 门控循环单元中隐藏状态的计算</div>


值得注意的是，更新门可以控制隐藏状态应该如何被包含当前时间步信息的候选隐藏状态所更新，如图3.3所示。假设更新门在时间步$t'$到$t$（$t' < t$）之间一直近似1。那么，在时间步$t'$到$t$之间的输入信息几乎没有流入时间步$t$的隐藏状态$\boldsymbol{H}_ t$。实际上，这可以看作是较早时刻的隐藏状态$\boldsymbol{H}_ {t'-1}$一直通过时间保存并传递至当前时间步$t$。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

我们对门控循环单元的设计稍作总结：
## 小结

* 门控循环神经网络可以更好地捕捉时间序列中时间步距离较大的依赖关系。
* 门控循环单元引入了门的概念，从而修改了循环神经网络中隐藏状态的计算方式。它包括重置门、更新门、候选隐藏状态和隐藏状态。
* 重置门有助于捕捉时间序列里短期的依赖关系。
* 更新门有助于捕捉时间序列里长期的依赖关系。
-----------
> [原书传送门](https://zh.d2l.ai/chapter_recurrent-neural-networks/gru.html)

## 3.2 长短期记忆LSTM（Long Short-term Memory）
本节将介绍另一种常用的门控循环神经网络：长短期记忆LSTM。它比门控循环单元的结构稍微复杂一点。

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息。

### 3.2.1 输入门、遗忘门和输出门

与门控循环单元中的重置门和更新门一样，如图3.4所示，长短期记忆的门的输入均为当前时间步输入$\boldsymbol{X}_ t$与上一时间步隐藏状态$\boldsymbol{H}_ {t-1}$，输出由激活函数为sigmoid函数的全连接层计算得到。如此一来，这3个门元素的值域均为$[0,1]$。

<div align=center>
<img width="500" src="image/task03/6.8_lstm_0.svg"/>
</div>
<div align=center>图3.4 长短期记忆中输入门、遗忘门和输出门的计算</div>

具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$\boldsymbol{X}_ t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$\boldsymbol{H}_ {t-1} \in \mathbb{R}^{n \times h}$。
时间步$t$的输入门$\boldsymbol{I}_ t \in \mathbb{R}^{n \times h}$、遗忘门$\boldsymbol{F}_ t \in \mathbb{R}^{n \times h}$和输出门$\boldsymbol{O}_ t \in \mathbb{R}^{n \times h}$分别计算如下：

$$
\begin{aligned}
\boldsymbol{I}_ t &= \sigma(\boldsymbol{X}_ t \boldsymbol{W}_ {xi} + \boldsymbol{H}_ {t-1} \boldsymbol{W}_ {hi} + \boldsymbol{b}_ i),
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol{F}_ t &= \sigma(\boldsymbol{X}_ t \boldsymbol{W}_ {xf} + \boldsymbol{H}_ {t-1} \boldsymbol{W}_ {hf} + \boldsymbol{b}_ f),
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol{O}_ t &= \sigma(\boldsymbol{X}_ t \boldsymbol{W}_ {xo} + \boldsymbol{H}_ {t-1} \boldsymbol{W}_ {ho} + \boldsymbol{b}_ o),
\end{aligned}
$$

其中的$\boldsymbol{W}_ {xi}, \boldsymbol{W}_ {xf}, \boldsymbol{W}_ {xo} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_ {hi}, \boldsymbol{W}_ {hf}, \boldsymbol{W}_ {ho} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_ i, \boldsymbol{b}_ f, \boldsymbol{b}_ o \in \mathbb{R}^{1 \times h}$是偏差参数。

### 3.2.2 候选记忆细胞

接下来，长短期记忆需要计算候选记忆细胞$\tilde{\boldsymbol{C}}_ t$。它的计算与上面介绍的3个门类似，但使用了值域在$[-1, 1]$的tanh函数作为激活函数，如图3.5所示。

<div align=center>
<img width="500" src="image/task03/6.8_lstm_1.svg"/>
</div>
<div align=center>图3.5 长短期记忆中候选记忆细胞的计算</div>


具体来说，时间步$t$的候选记忆细胞$\tilde{\boldsymbol{C}}_ t \in \mathbb{R}^{n \times h}$的计算为

$$
\tilde{\boldsymbol{C}}_ t = \text{tanh}(\boldsymbol{X}_ t \boldsymbol{W}_ {xc} + \boldsymbol{H}_ {t-1} \boldsymbol{W}_ {hc} + \boldsymbol{b}_ c),
$$

其中$\boldsymbol{W}_ {xc} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_ {hc} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_ c \in \mathbb{R}^{1 \times h}$是偏差参数。


### 3.2.3 记忆细胞

我们可以通过元素值域在$[0, 1]$的输入门、遗忘门和输出门来控制隐藏状态中信息的流动，这一般也是通过使用按元素乘法（符号为$\odot$）来实现的。当前时间步记忆细胞$\boldsymbol{C}_ t \in \mathbb{R}^{n \times h}$的计算组合了上一时间步记忆细胞和当前时间步候选记忆细胞的信息，并通过遗忘门和输入门来控制信息的流动：

$$\boldsymbol{C}_ t = \boldsymbol{F}_ t \odot \boldsymbol{C}_ {t-1} + \boldsymbol{I}_ t \odot \tilde{\boldsymbol{C}}_ t.$$


如图3.6所示，遗忘门控制上一时间步的记忆细胞$\boldsymbol{C}_ {t-1}$中的信息是否传递到当前时间步，而输入门则控制当前时间步的输入$\boldsymbol{X}_ t$通过候选记忆细胞$\tilde{\boldsymbol{C}}_ t$如何流入当前时间步的记忆细胞。如果遗忘门一直近似1且输入门一直近似0，过去的记忆细胞将一直通过时间保存并传递至当前时间步。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

<div align=center>
<img width="500" src="image/task03/6.8_lstm_2.svg"/>
</div>
<div align=center>图3.6 长短期记忆中记忆细胞的计算</div>


### 3.2.4 隐藏状态

有了记忆细胞以后，接下来我们还可以通过输出门来控制从记忆细胞到隐藏状态$\boldsymbol{H}_ t \in \mathbb{R}^{n \times h}$的信息的流动：

$$\boldsymbol{H}_ t = \boldsymbol{O}_ t \odot \text{tanh}(\boldsymbol{C}_ t).$$

这里的tanh函数确保隐藏状态元素值在-1到1之间。需要注意的是，当输出门近似1时，记忆细胞信息将传递到隐藏状态供输出层使用；当输出门近似0时，记忆细胞信息只自己保留。图3.7展示了长短期记忆中隐藏状态的计算。

<div align=center>
<img width="500" src="image/task03/6.8_lstm_3.svg"/>
</div>
<div align=center>图3.7 长短期记忆中隐藏状态的计算</div>

## 小结

* 长短期记忆的隐藏层输出包括隐藏状态和记忆细胞。只有隐藏状态会传递到输出层。
* 长短期记忆的输入门、遗忘门和输出门可以控制信息的流动。
* 长短期记忆可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

-----------
> [LSTM](https://zhuanlan.zhihu.com/p/32085405)
> [原书传送门](https://zh.d2l.ai/chapter_recurrent-neural-networks/lstm.html)

## 3.3 深度循环神经网络DRNN（Deep Recurrent Neural Network）

## 3.4 双向循环神经⽹络BRNN（Bidirectional Recurrent Neural Network）



