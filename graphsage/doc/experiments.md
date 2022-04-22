

## DGL-0.1.x

### rgcn（无采样）

| 数据集 | 节点数  | 边数     | 相关数 | 耗时(前向/反向)         | 内存消耗 | 备注                                            |
| ------ | ------- | -------- | ------ | ----------------------- | -------- | ----------------------------------------------- |
| BGS    | 333845  | 2166243  | 207    | 0.106217<br />0.229685  | 3.2G     | 观察到最大3.2G，后来平稳在2.547G<br />显存6589M |
| MUTAG  | 23644   | 172098   | 47     | 0.007904 <br />0.017471 | 2.254G   | 显存 1573M                                      |
| AIFB   | 8285    | 66371    | 91     | 0.004371 <br />0.008215 | 2.232G   | 显存1413M                                       |
| AM     | 1666764 | 13643406 | 267    |                         |          | 显存爆了                                        |

rgcn的文档里给的示例是`bgs、mutag、aifb`三个数据集，正常跑没问题，`0.8.x`版本里新增了`am`数据集，尝试在低版本跑了下，显存爆了



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

### gat

```bash
conda activate dgl-0.1.x
cd /workspace/0.1.x/examples/pytorch/gat
python gat.py --dataset cora --gpu 0 --num-heads 8 --epochs 10   # dataset可选"cora", "pubmed", "citeseer"
```



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

```bash
conda activate dgl-0.7.x
cd /workspace/0.7.x/examples/pytorch/graphsage

python3 train_full.py --dataset reddit --gpu 0 --aggregator-type gcn --n_epochs 200
# -- 输出信息--
# Namespace(aggregator_type='gcn', dataset='reddit', dropout=0.5, gpu=0, lr=0.01, n_epochs=200, n_hidden=16, n_layers=1, weight_decay=0.0005)
Test Accuracy 0.9329

python3 train_full.py --dataset reddit --gpu 0 --aggregator-type mean --n-epochs 200
# Namespace(aggregator_type='mean', dataset='reddit', dropout=0.5, gpu=0, lr=0.01, n_epochs=200, n_hidden=16, n_layers=1, weight_decay=0.0005)
Test Accuracy 0.9448

```

```bash
conda activate dgl-0.7.x
pip install tqdm sklearn
cd /workspace/0.7.x/examples/pytorch/graphsage

python3 train_sampling.py --dataset reddit --num-workers 0 --gpu 0  

# -- 输出信息--
# Namespace(batch_size=1000, data_cpu=False, dataset='reddit', dropout=0.5, eval_every=5, fan_out='10,25', gpu=0, inductive=False, log_every=20, lr=0.003, num_epochs=20, num_hidden=16, num_layers=2, num_workers=0, sample_gpu=False)
Epoch Time(s): 5.3352
Eval Acc 0.9472
```



### RGAT

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



### DGL-0.8.x

#### rgcn

`DGL两版本算法采样对比`

|      | DGL0.8.x | DGL0.1.x |
| ---- | -------- | -------- |
| gcn  | 无采样   | 无采样   |
| gat  | 无采样   | 无采样   |
| rgcn | 两者都有 | 无采样   |
| rgat | 有采样   | ——       |

am

4689M

rgcn-hetero

12125M



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

