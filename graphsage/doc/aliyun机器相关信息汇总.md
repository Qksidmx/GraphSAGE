# docker镜像

## opeceipeno/graphsage:gpu

graphSAGE的环境，[GitHub地址](https://github.com/qksidmx/GraphSAGE)

```bash
docker run --gpus all -it opeceipeno/graphsage:gpu bash
#/notebook目录下面有代码，运行实验参考readme文档

# 启动命令参考
python -m graphsage.supervised_train \
--train_prefix ./ppi/ppi \  #数据集目录，全量数据可选./ppi/ppi 和./reddit/reddit，测试数据可选./example_data/toy-ppi
--model graphsage_mean \
--learning_rate 0.001 \
--epochs 200 \
--batch_size 1024 \
--validate_iter 10 \
--random_context False \
--sigmoid \
--gpu 0
```

-----



## opeceipeno/dgl:devel-gpu

### 启动镜像命令

```bash
docker run --gpus all -ti opeceipeno/dgl:devel-gpu bash
```

### `conda`虚拟环境列表

```bash
conda env list

#----输出----#
base                     /opt/conda
mxnet-ci                 /opt/conda/envs/mxnet-ci
pytorch-ci               /opt/conda/envs/pytorch-ci
tensorflow-ci            /opt/conda/envs/tensorflow-ci
torch-0.1.x           *  /opt/conda/envs/torch-0.1.x
```

### 数据集

在`/root/.dgl`文件夹中存放了`cora`, `pubmed`, `citeseer`三个数据集，因为`dgl-0.1.x`版本较老，已经无法通过该包下载数据集了，只能手动下载放入相应的目录，才能够测试。

```bash
tree /root/.dgl

#---------#
/root/.dgl/
|-- citeseer
|   |-- ind.citeseer.allx
|   |-- ind.citeseer.ally
|   |-- ind.citeseer.graph
|   |-- ind.citeseer.test.index
|   |-- ind.citeseer.tx
|   |-- ind.citeseer.ty
|   |-- ind.citeseer.x
|   `-- ind.citeseer.y
|-- citeseer.zip
|-- cora      # 该数据集是cora第一版，只有.cites和.content两个文件，dgl的0.1.x版本的gcn示例使用的是这个结构的数据集
|   |-- README
|   |-- cora.cites
|   `-- cora.content
|-- cora.zip  # 该版本即为第一版数据集结构
|-- cora_v2.zip   # 这是cora第二版数据集，未解压，里面的文件结构是形如pubmed和citeseer的，新版dgl加载cora的话是用的这个版本
|-- pubmed
|   |-- ind.pubmed.allx
|   |-- ind.pubmed.ally
|   |-- ind.pubmed.graph
|   |-- ind.pubmed.test.index
|   |-- ind.pubmed.tx
|   |-- ind.pubmed.ty
|   |-- ind.pubmed.x
|   `-- ind.pubmed.y
`-- pubmed.zip

```

### 编译安装`dgl`

#### 0.1.x

```bash
# 激活pytorch的编译环境
conda activate torch-0.1.x

# 进入目录并编译
mkdir /workspace/0.1.x/build && cd /workspace/0.1.x/build
cmake .. 
make -j4

# 安装pip包
cd ../python
python setup.py install
```

测试：

```bash
# 测试
cd /workspace/0.1.x/examples/pytorch/gcn
python gcn_spmv.py --dataset cora --gpu 0   # dataset可选"cora", "pubmed", "citeseer"
```

更多信息请看后文的实验部分

#### 0.7.x

```bash
# 激活pytorch的编译环境
conda activate pytorch-ci

# 进入目录并编译
mkdir /workspace/0.7.x/build && cd /workspace/0.7.x/build
cmake -DUSE_CUDA=ON -DBUILD_TORCH=ON ..
make -j4

# 安装pip包
cd ../python
python setup.py install
```

测试：

```bash
python -c "import dgl; print(dgl.__version__);import torch; print(torch.cuda.is_available())"

# 一个官方的示例脚本
cd /workspace && python dgl_introduction-gpu.py
```

---

### opeceipeno/dgl:v1.4

ogb代码的运行环境，想法是通过虚拟环境去激活各个方案的运行环境，当前做好了Google的mag240m运行环境

[GitHub地址](https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc/mag)

```
docker run --gpus all -it -v /mnt:/mnt opeceipeno/dgl:v1.4 bash
# 启动容器后，激活Google代码的运行环境
source /py3_venv/google_ogb_mag240m/bin/activate
# /workspace 目录有代码
```

Google方案预处理后的数据目录：`/mnt/ogb-dataset/mag240m/data/preprocessed`，相当于执行完了`run_preprocessing.sh`脚本，下一步是可以复现实验，







# 实验

## dgl 0.1.x

请先确保在`conda torch-0.1.x `虚拟环境下，并编译安装好了dgl包

### gcn

```bash
# 测试
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
# 测试
cd /workspace/0.1.x/examples/pytorch/gat
python gat.py --dataset cora --gpu 0 --num-heads 8 --epochs 10   # dataset可选"cora", "pubmed", "citeseer"
```



### rgcn

这个用的是abfi数据集，因为版本太老，dgl内置下载数据集的网址连不上了下载不了，先搁着





## dgl 0.7.x

请先确保在`conda pytorch-ci `虚拟环境下，并编译安装好了dgl包

### graphSAGE

完整文档参考https://github.com/dmlc/dgl/tree/0.7.x/examples/pytorch/graphsage

```bash
# 有监督训练
cd /workspace/0.7.x/examples/pytorch/graphsage
python3 train_full.py --dataset cora --gpu 0


Epoch 00197 | Time(s) 0.0053 | Loss 0.3799 | Accuracy 0.8020 | ETputs(KTEPS) 2007.47
Epoch 00198 | Time(s) 0.0053 | Loss 0.3787 | Accuracy 0.8020 | ETputs(KTEPS) 2008.70
Epoch 00199 | Time(s) 0.0052 | Loss 0.3889 | Accuracy 0.8020 | ETputs(KTEPS) 2011.74

Test Accuracy 0.8170
```



无监督训练因为个人PC内存不够载入不了reddit数据集，等阿里云机器空闲了再补上



# MAG240M数据集

阿里云IP：敏感信息不放在网上

### mag240原数据

目录：`/mnt/ogb-dataset/mag240m/data/raw`

```
├── RELEASE_v1.txt
├── mapping    //空文件夹
├── meta.pt
├── processed
│   ├── author___affiliated_with___institution
│   │   └── edge_index.npy     //作者和机构的边，shape=[2,num_edges]
│   ├── author___writes___paper
│   │   └── edge_index.npy     //作者和论文的边，shape=[2,num_edges]
│   ├── paper
│   │   ├── node_feat.npy    //论文节点的特征，shape=[num_node,768]
│   │   ├── node_label.npy   // 论文的标签
│   │   └── node_year.npy   // 论文年份
│   └── paper___cites___paper
│       └── edge_index.npy  // 论文引用关系的边shape=[2,num_edges]
├── raw //空文件夹
└── split_dict.pt    //切分训练集、验证集、测试集方式的文件，用torch读取是一个字典，keys=[‘train’,’valid’,’test’], value是node_index

```



```python
from ogb.lsc import MAG240MDataset

dataset = MAG240MDataset(root = “/mnt/ogb-dataset/mag240m/data/raw”)
# 使用请参考 https://ogb.stanford.edu/docs/lsc/mag240m/
```







### 按年切分的数据集

目录：`/mnt/ogb-dataset/mag240m/data/split_data_by_year`，文件较多，以`1960-1969_paper_nodes.npz`为例

```python
import numpy as np 

data = np.load("1960-1969_paper_nodes.npz")

feats = data["feats"]  # 节点特征
year = data["year"] # 节点年份
idx = data["idx"] #节点在原始数据集的index
labels = data["labels"] # 节点标签
```



