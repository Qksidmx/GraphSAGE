## DGL0.1.x原理拆解

### dgl编译

克隆仓库，设置分支：

```
git clone https://github.com/dmlc/dgl
git checkout -b 0.1.x --track origin/0.1.x
```

依赖dlpack和dmlc-core库，编译前需更新

```
git submodule update --init --recursive
```

编译流程：

```
# 后续若未指明执行路径的指令，均为dgl仓库的根路径
mkdir output && cd output
cmake ..
make
sudo make install
```

编译安装完毕后，即可得到libdgl.so。linux下默认安装位置为/usr/local/lib/libdgl.so

python编译流程

```
cp output/libdgl.so lib
cd python && python3 setup.py build
```

python文件将生成在python/build/ 目录下

引用了libdgl.so的文件：

```
./python/dgl/_ffi/libinfo.py
./python/dgl/_ffi/base.py
```

### dgl调试

建议使用docker镜像或virtualenv/conda等虚拟环境调试python

```
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
```

dgl的python库依赖：

```
networkx==2.1
torch==1.0.0
numpy
scipy
```

按编译章节中的步骤build好dgl的python库之后，
将so放入对应文件夹下，即可在build目录下调试

```
cp output/libdgl.so python/build/lib/
```

设置python的库path，方便import dgl

```
import sys
sys.path.append('/data/jiangjiajun750/cpp/dgl/python/build/lib')
import dgl
dgl.__version__
```

执行单元测试：

```
cd tests/graph_index && python3 test_basics.py
```

#### docker环境搭建

##### dockerfile

```dockerfile
#####################
##  dgl开发环境镜像  ##
#####################

# 基础镜像选的英伟达11.2.2的cuda环境
FROM nvidia/cuda:11.2.2-devel-ubuntu18.04

# 拷贝相关文件，dep目录下的文件见后文
COPY ./dep /tmp



# 安装基础软件
RUN apt-get update && \ 
    apt-get install --no-install-recommends build-essential python3-dev make wget vim -y && \
    apt-get clean && \
    echo y | /bin/bash /tmp/cmake-3.22.3-linux-x86_64.sh --prefix=/usr/local && \
	/bin/bash /tmp/miniconda3.sh -b -p /opt/conda && \
	mkdir -p /root/.dgl /workspace && \
	ls /tmp/dgl-src/*.tar.gz | xargs -n1 -i tar zxf {} -C /workspace && \
	mv /tmp/dgl_tutorial/dgl_introduction-gpu.py /workspace/. && \
	mv /tmp/dataset/* /root/.dgl/. && \
	dpkg -i /tmp/cudnn/*.deb && \
	rm -rf /var/lib/apt/lists/* /root/.cache/* /tmp/*

# Put conda and cmake in path
ENV CONDA_DIR=/opt/conda \
	CMAKE_DIR=/usr/local/cmake-3.22.3-linux-x86_64

ENV PATH=$CONDA_DIR/bin:$CMAKE_DIR/bin:$PATH

# init conda and create conda env
RUN conda init bash && \
	conda create --name dgl-0.1.x python=3.8 -y && \ 
	conda create --name dgl-0.7.x python=3.8 -y

# 切换到0.1.x环境并安装相关包
SHELL ["conda", "run", "-n", "dgl-0.1.x", "/bin/bash", "-c"]

RUN python -m pip install \
	networkx==2.1 \
	scipy==1.8.0 \
	torch==1.4.0 \
	matplotlib==3.1.3 \
	requests \
	--no-cache-dir \
	-i https://pypi.tuna.tsinghua.edu.cn/simple

# 切换到0.7.x环境并安装相关包
SHELL ["conda", "run", "-n", "dgl-0.7.x", "/bin/bash", "-c"]
RUN python -m pip install \
	networkx==2.5.1 \
	scipy==1.8.0 requests \
	--no-cache-dir \
	-i https://pypi.tuna.tsinghua.edu.cn/simple && \
	python -m pip install \
	torch==1.9.0+cu111 \
	torchvision==0.10.0+cu111 \
	torchaudio==0.9.0 \
	--no-cache-dir \
	-f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
```

##### dep文件目录

```bash
dep，
├── cmake-3.22.3-linux-x86_64.sh  # 新版cmake， 通过apt-get安装的cmake 是3.10.2版本，在编译CUDA的时候会出现问题，需要用较新版本的解决
├── cudnn # cudnn安装包
│   ├── libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
│   ├── libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb
│   └── libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
├── dataset # 数据集
│   ├── citeseer
│   │   ├── ind.citeseer.allx
│   │   ├── ind.citeseer.ally
│   │   ├── ind.citeseer.graph
│   │   ├── ind.citeseer.test.index
│   │   ├── ind.citeseer.tx
│   │   ├── ind.citeseer.ty
│   │   ├── ind.citeseer.x
│   │   └── ind.citeseer.y
│   ├── cora
│   │   ├── README
│   │   ├── cora.cites
│   │   └── cora.content
│   ├── cora_v2
│   │   ├── ind.cora_v2.allx
│   │   ├── ind.cora_v2.ally
│   │   ├── ind.cora_v2.graph
│   │   ├── ind.cora_v2.test.index
│   │   ├── ind.cora_v2.tx
│   │   ├── ind.cora_v2.ty
│   │   ├── ind.cora_v2.x
│   │   └── ind.cora_v2.y
│   └── pubmed
│       ├── ind.pubmed.allx
│       ├── ind.pubmed.ally
│       ├── ind.pubmed.graph
│       ├── ind.pubmed.test.index
│       ├── ind.pubmed.tx
│       ├── ind.pubmed.ty
│       ├── ind.pubmed.x
│       └── ind.pubmed.y
├── dgl-src  # 代码，打成tar包了
│   ├── dgl-src-0.1.x.tar.gz
│   └── dgl-src-0.7.x.tar.gz 
├── dgl_tutorial # dgl简单的示例
│   ├── README.md
│   └── dgl_introduction-gpu.py
└── miniconda3.sh  # miniconda3安装脚本
```





### 函数注册流程分析

python调用方之一：`./python/dgl/graph_index.py`，调用方式如下：

```
handle = _CAPI_DGLGraphCreate(multigraph)
gi = GraphIndex(handle)
```

C函数：`_CAPI_DGLGraphCreate`，cc文件在`./src/graph/graph_apis.cc`

```
58 DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCreate")
59 .set_body([] (DGLArgs args, DGLRetValue* rv) {
60     bool multigraph = static_cast<bool>(args[0]);
61     GraphHandle ghandle = new Graph(multigraph);
62     *rv = ghandle;
63   });
```

C接口注册宏（`DGL_REGISTER_GLOBAL`）定义在：`./include/dgl/runtime/registry.h`
用于定义DGL全局函数

```
#define DGL_REGISTER_GLOBAL(OpName)                              \
  DGL_STR_CONCAT(DGL_FUNC_REG_VAR_DEF, __COUNTER__) =            \
      ::dgl::runtime::Registry::Register(OpName)
```

设置函数体的方法：

```
42 Registry& Registry::set_body(PackedFunc f) {
43   func_ = f;
44   return *this;
45 }
```

python注册c函数流程分析：

此文件`./python/dgl/_ffi/function.py` 将已注册到fmap中的c函数注册到每个python的module中，因此每个module不需要显示声明c函数即可调用

### 文件结构

```
./include/dgl/graph.h
./include/dgl/runtime/registry.h 定义DGL全局函数注册，供前端和后端用户使用
./include/dgl/runtime/serializer.h
./include/dgl/runtime/threading_backend.h
./include/dgl/runtime/module.h
./include/dgl/runtime/c_backend_api.h
./include/dgl/runtime/packed_func.h 定义DGL API的类型擦除函数，提供调用匿名函数的统一接口，包含类：DGLArgs/DGLArgValue/DGLRetValue/DGLArgsSetter
./include/dgl/runtime/device_api.h
./include/dgl/runtime/util.h
./include/dgl/runtime/c_runtime_api.h 定义DGL运行时的库，例如DGLValue结构
./include/dgl/runtime/ndarray.h
./include/dgl/scheduler.h
./include/dgl/graph_op.h

./src/graph/traversal.h
./src/runtime/file_util.h
./src/runtime/thread_storage_scope.h
./src/runtime/module_util.h
./src/runtime/meta_data.h
./src/runtime/pack_args.h
./src/runtime/runtime_base.h
./src/runtime/workspace_pool.h
./src/c_api_common.h

./src/graph/graph.cc
./src/graph/traversal.cc
./src/graph/graph_op.cc
./src/graph/graph_apis.cc
./src/runtime/ndarray.cc
./src/runtime/threading_backend.cc
./src/runtime/module_util.cc
./src/runtime/thread_pool.cc
./src/runtime/cpu_device_api.cc
./src/runtime/c_runtime_api.cc
./src/runtime/registry.cc
./src/runtime/workspace_pool.cc
./src/runtime/system_lib_module.cc
./src/runtime/dso_module.cc
./src/runtime/module.cc
./src/runtime/file_util.cc
./src/scheduler/scheduler.cc
./src/scheduler/scheduler_apis.cc
./src/c_api_common.cc
```

### 参考：

* openmp多线程编程：https://blog.csdn.net/acaiwlj/article/details/49818965
