

## DGL-0.1.x

`DGL两版本算法采样对比`

|      | DGL0.8.x | DGL0.1.x |
| ---- | -------- | -------- |
| gcn  | 无采样   | 无采样   |
| gat  | 无采样   | 无采样   |
| rgcn | 两者都有 | 无采样   |
| rgat | 有采样   | ——       |

### rgcn（无采样）

| 数据集 | 节点数  | 边数     | 相关数 | 耗时(前向/反向)         | 内存消耗 | 备注                                            |
| ------ | ------- | -------- | ------ | ----------------------- | -------- | ----------------------------------------------- |
| BGS    | 333845  | 2166243  | 207    | 0.106217<br />0.229685  | 3.2G     | 观察到最大3.2G，后来平稳在2.547G<br />显存6589M |
| MUTAG  | 23644   | 172098   | 47     | 0.007904 <br />0.017471 | 2.254G   | 显存 1573M                                      |
| AIFB   | 8285    | 66371    | 91     | 0.004371 <br />0.008215 | 2.232G   | 显存1413M                                       |
| AM     | 1666764 | 13643406 | 267    |                         |          | 显存爆了                                        |

`0.1.x rgcn`的文档里给的示例是`bgs、mutag、aifb`三个数据集，正常跑没问题，`0.8.x`版本里新增了`am`数据集，尝试下载数据集之后拿到`0.1.x`版本跑了下，显存爆了，高版本可正常跑，有可能旧版本这块儿没优化好。



假设一个图包含`N`个节点，每个节点的特征维度为`H`，使用`L`层的`GCN`训练的话，那么存储隐藏层需要的内存为*`O(NHL)`*，N较大时，GPU显存就会不够。

*参考[Chapter 6: Stochastic Training on Large Graphs — DGL 0.8.1 documentation](https://docs.dgl.ai/en/0.8.x/guide/minibatch.html)*





#### 命令

```bash
conda activate dgl-0.1.x
cd /workspace/0.1.x/examples/pytorch/rgcn

python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --relabel
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
python3 entity_classify.py -d aifb --testing --gpu 0
python3 entity_classify.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

### gcn

```bash
conda activate dgl-0.1.x
cd /workspace/0.1.x/examples/pytorch/gcn
python gcn_spmv.py --dataset cora --gpu 0   # dataset可选"cora", "pubmed", "citeseer"
```

以下是自测结果，跟论文结果差不多

| 数据集     | 结果(准确度) |
| ---------- | ------------ |
| `cora`     | 0.8270       |
| `pubmed`   | 0.7950       |
| `citeseer` | 0.6950       |

#### 加载ogbn_mag数据，运行实验

`dgl 0.1.x`版本不支持异构图，因此此次实验只加载了mag数据集的论文节点图做训练，相关代码已上传至git，使用full batch训练资源消耗情况如下：

|      | 论文节点数 | 论文引用边数 | 节点特征维度 | hidden维度 | 内存消耗 | 显存消耗 |      |
| ---- | ---------- | ------------ | ------------ | ---------- | -------- | -------- | ---- |
|      | 79W        | 570W         | 128          | 16         | 3.19G    | 13G      |      |

### 

### gat

```bash
conda activate dgl-0.1.x
cd /workspace/0.1.x/examples/pytorch/gat
python gat.py --dataset cora --gpu 0 --num-heads 8 --epochs 10   # dataset可选"cora", "pubmed", "citeseer"
```



## DGL-0.8.x

`ogbn-mag`数据集信息

|          | 节点数    | 边数       |      |
| -------- | --------- | ---------- | ---- |
| ogbn-mag | 1,939,743 | 21,111,007 |      |

### rgcn 

(代码目录`examples/pytorch/rgcn`)

#### 数据集介绍

```python
import dgl
dataset = dgl.data.AIFBDataset()
dataset = dgl.data.MUTAGDataset()
dataset = dgl.data.BGSDataset()
dataset = dgl.data.AMDataset()
```

使用上述`DGL0.8.x`的内置数据集加载，最后得到的图和原始的略有不同。 DGL直接加载和预处理原始 RDF 数据。 对于 AIFB、BGS 和 AM，从图中删除所有文字节点。 对于 AIFB，从训练/测试集中排除孤立节点。最终生成的图的节点、边、关系数目都有减少。下图是区别(带有`-hetero`的是DGL的) ：

| Dataset      | #Nodes    | #Edges     | #Relations | #Labeled | 备注 |
| ------------ | --------- | ---------- | ---------- | -------- | ---- |
| AIFB         | 8,285     | 58,086     | 90         | 176      |      |
| AIFB-hetero  | 7,262     | 48,810     | 78         | 176      |      |
| MUTAG        | 23,644    | 148,454    | 46         | 340      |      |
| MUTAG-hetero | 27,163    | 148,100    | 46         | 340      |      |
| BGS          | 333,845   | 1,832,398  | 206        | 146      |      |
| BGS-hetero   | 94,806    | 672,884    | 96         | 146      |      |
| AM           | 1,666,764 | 11,976,642 | 266        | 1000     |      |
| AM-hetero    | 881,680   | 5,668,682  | 96         | 1000     |      |

表格和数据集描述均摘自`DGL`，下面的实验都是基于`-hetero`数据集跑的，但`rgcn`模型并没用做异构图的特殊处理

#### FULL-BATCH

**运行指令参考**

```bash
python entity.py -d aifb --wd 0 --gpu 0
python entity.py -d mutag --n-bases 30 --gpu 0
python entity.py -d bgs --n-bases 40 --gpu 0
python entity.py -d am --n-bases 40 --gpu 0
```



**实验结果(运行5次取平均)**

| Dataset      | 准确度                            | 耗时(前向/反向)         | 内存消耗 | 显存消耗 | 备注 |
| ------------ | --------------------------------- | ----------------------- | -------- | -------- | ---- |
| AIFB-hetero  | 0.8611/**0.9056**(无/有 selfloop) | 0.007415<br /> 0.004401 | 4.855G   | 1773M    |      |
| MUTAG-hetero | **0.6824**/0.6030                 | 0.013175<br /> 0.003879 | 4.872G   | 1845M    |      |
| BGS-hetero   | **0.8827**/0.8759                 | 0.048400<br /> 0.006302 | 4.868G   | 2105M    |      |
| AM-hetero    | **0.8162**/0.8060                 | 0.421278<br /> 0.007212 | 4.925G   | 5501M    |      |

两个结果分别代表不使用`self_loop`和使用`self_loop`的效果，除了AIFB数据集，其他的数据集使用`self_loop`之后效果降低了。





#### MINI-BATCH

**运行指令**

```bash
# 默认采样4,4
python entity_sample.py -d aifb --wd 0 --gpu 0  --batch-size 128
python entity_sample.py -d mutag --n-bases 30 --gpu 0 --batch-size 64
python entity_sample.py -d bgs --n-bases 40 --gpu 0 --batch-size=16
python entity_sample.py -d am --n-bases 40 --gpu 0 --batch-size 64 

#采样15,25
python entity_sample.py -d aifb --wd 0 --gpu 0  --batch-size 128 --use-self-loop  --fanout=15,25 
python entity_sample.py -d mutag --n-bases 30 --gpu 0 --batch-size 64 --use-self-loop --fanout=15,25
python entity_sample.py -d bgs --n-bases 40 --gpu 0 --batch-size=16 --use-self-loop --fanout=15,25     
python entity_sample.py -d am --n-bases 40 --gpu 0 --batch-size 64 --use-self-loop  --fanout=15,25
```



**实验结果(运行5次取平均)**

| Dataset      | 采样：4,4                          | 采样：15,25 (有self_loop) | 备注 |
| ------------ | ---------------------------------- | ------------------------- | ---- |
| AIFB-hetero  | 0.8224/**0.8500** (无/有 selfloop) | 0.8611                    |      |
| MUTAG-hetero | 0.5617/**0.6471**                  | 0.6559                    |      |
| BGS-hetero   | **0.8027**/0.6759                  | 0.7724                    |      |
| AM-hetero    | 0.4071/**0.5273**                  | 0.7465                    |      |

如果使用`MINI-BATCH`，`self_loop`有一定效果，采样数增加，对效果有改善，但仍然比`FULL-BATCH`效果差



#### 结论

从实验来看，`FULL-BATCH`的效果明显好于`MINI-BATCH`



### rgcn-hetero

 (代码目录`examples/pytorch/rgcn-hetero`)

| Dataset      | #Nodes    | #Edges     | #Relations | #Labeled | 内存消耗 | 显存消耗 | Acc        |
| ------------ | --------- | ---------- | ---------- | -------- | -------- | -------- | ---------- |
| AIFB         | 8,285     | 58,086     | 90         | 176      |          |          |            |
| AIFB-hetero  | 7,262     | 48,810     | 78         | 176      | 4.502G   | 1671M    |            |
| MUTAG        | 23,644    | 148,454    | 46         | 340      |          |          |            |
| MUTAG-hetero | 27,163    | 148,100    | 46         | 340      | 4.494G   | 1689M    |            |
| BGS          | 333,845   | 1,832,398  | 206        | 146      |          |          |            |
| BGS-hetero   | 94,806    | 672,884    | 96         | 146      | 4.506G   | 1805M    |            |
| AM           | 1,666,764 | 11,976,642 | 266        | 1000     |          |          |            |
| AM-hetero    | 881,680   | 5,668,682  | 96         | 1000     | 4.634G   | 12125M   | **0.8230** |



**运行指令**

```bash
python3 entity_classify.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

#### 结论

rgcn-hetero比rgcn性能好些，也占用了更多的资源，具体模型结构还没看



## DGL-0.7.x

### graphSAGE

#### cora

完整文档参考https://github.com/dmlc/dgl/tree/0.7.x/examples/pytorch/graphsage

```bash
conda activate dgl-0.7.x
# 有监督训练
cd /workspace/0.7.x/examples/pytorch/graphsage
python3 train_full.py --dataset cora --gpu 0 

Test Accuracy 0.8170
```

#### reddit

文档里跑的结果如下，`FULL-BATCH`的效果略好于`MINI-BATCH`，文档没说明参数

| Model             | Accuracy |
| ----------------- | -------- |
| Full Graph        | 0.9504   |
| Neighbor Sampling | 0.9495   |



以下是自己跑的结果

```bash
conda activate dgl-0.7.x
cd /workspace/0.7.x/examples/pytorch/graphsage

python3 train_full.py --dataset reddit --gpu 0 --aggregator-type gcn --n-epochs 200
# -- 输出信息--
# Namespace(aggregator_type='gcn', dataset='reddit', dropout=0.5, gpu=0, lr=0.01, n_epochs=200, n_hidden=16, n_layers=1, weight_decay=0.0005)
# 注意这里n_layers=1， 表示的是模型隐藏层层数，但是实际上在代码里，模型构建只会有个输入层和输出层，隐藏层为0
Test Accuracy 0.9340

python3 train_full.py --dataset reddit --gpu 0 --aggregator-type mean --n-epochs 200
# Namespace(aggregator_type='mean', dataset='reddit', dropout=0.5, gpu=0, lr=0.01, n_epochs=200, n_hidden=16, n_layers=1, weight_decay=0.0005)
Test Accuracy 0.9481

```

```bash
conda activate dgl-0.7.x
pip install tqdm sklearn
cd /workspace/0.7.x/examples/pytorch/graphsage

python3 train_sampling.py --dataset reddit --num-workers 0 --gpu 0  --lr 0.01 --num-epochs=200

# -- 输出信息--
# 采样的聚合方式代码里固定是 mean
# 注意这里num_layers=2， 表示的是模型隐藏层层数，但是实际上在代码里，模型只会有个输入层和输出层，隐藏层为0
# Namespace(batch_size=1000, data_cpu=False, dataset='reddit', dropout=0.5, eval_every=5, fan_out='10,25', gpu=0, inductive=False, log_every=20, lr=0.01, num_epochs=200, num_hidden=16, num_layers=2, num_workers=0, sample_gpu=False)
Epoch Time(s): 5.3352
Test Acc: 0.9492

# 在20个epoch的时候，Acc就基本收敛了，比full batch收敛速度快很多
```



### RGAT

 (代码目录`examples/pytorch/ogb_lsc/MAG240M`)

该目录是dgl的MAG240M示例代码，使用的是RGAT模型，我根据代码，将数据集换成了reddit测试效果。

#### reddit

节点数：232965  

边数：114615892

节点特征数：602

| fan-out (9999代表所有邻居都采) | 显存  |
| ------------------------------ | ----- |
| 9999                           | 11.2G |
| 10,25                          | 10.4G |
| 9999,9999                      | 爆了  |
| 10,50                          | 13.1G |
| 999,999                        | 爆了  |



```bash
conda activate dgl-0.7.x
# 有监督训练
cd /workspace/0.7.x/examples/pytorch/ogb_lsc/MAG240M
python reddit_train.py --fan-out 10,25
# 数据集结果
Validation accuracy: 0.9680668037430238
Test accuracy: 0.9676139525698796
```



效果比graphSAGE好2%。







## GraphSAGE论文官方代码

### 有监督

```bash
python -m graphsage.supervised_train \
--train_prefix ./ppi/ppi \
--model graphsage_maxpool \
--learning_rate 0.001 \
--epochs 200 \
--batch_size 1024  \
--validate_iter 10 \
--random_context False \
--sigmoid \
--gpu 0

Full validation stats: loss= 0.35714 f1_micro= 0.71096 f1_macro= 0.63525 time= 0.30733
```



```bash
python -m graphsage.supervised_train \
--train_prefix ./ppi/ppi \
--model gcn \
--print_every 50 \
--epochs 200 \
--batch_size 1024 \
--learning_rate 0.001 \
--validate_iter 10 \
--random_context False \
--sigmoid \
--gpu 0


Full validation stats: loss= 0.49029 f1_micro= 0.52777 f1_macro= 0.34071 time= 0.25776
```



```bash
python -m graphsage.supervised_train \
--train_prefix ./ppi/ppi \
--model graphsage_seq \
--learning_rate 0.001 \
--epochs 200 \
--batch_size 1024 \
--validate_iter 10 \
--random_context False \
--sigmoid \
--gpu 0

Full validation stats: loss= 0.34045 f1_micro= 0.72752 f1_macro= 0.65601 time= 0.41571
```



```bash
python -m graphsage.supervised_train \
--train_prefix ./ppi/ppi \
--model graphsage_mean \
--learning_rate 0.001 \
--epochs 200 \
--batch_size 1024 \
--validate_iter 10 \
--random_context False \
--sigmoid \
--gpu 0

Full validation stats: loss= 0.42566 f1_micro= 0.60157 f1_macro= 0.47732 time= 0.30706
```



### 无监督

```bash
python -m graphsage.unsupervised_train \
--train_prefix ./ppi/ppi \
--model graphsage_mean \
--model_size big \
--print_every 50 \
--epoch 50 \
--batch_size 1024 \
--dropout 0.1 \
--learning_rate 0.0001 \
--validate_iter 10 \
--random_context False
--gpu 0


python eval_scripts/ppi_eval.py  ./ppi ./unsup-ppi/graphsage_mean_big_0.000100 test

F1-micro 0.761944872
```

