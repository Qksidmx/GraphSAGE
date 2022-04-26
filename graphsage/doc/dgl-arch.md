## DGL0.1.x原理拆解

- [dgl编译](#dgl编译)
- [dgl调试](#dgl调试)
  * [普通环境](#普通环境)
  * [docker环境](#docker环境)
    + [Dockerfile.devel-gpu](#dockerfiledevel-gpu)
    + [dep文件目录](#dep文件目录)
    + [Dockerfile.devel-gpu-compiled](#dockerfiledevel-gpu-compiled)
- [函数注册流程分析](#函数注册流程分析)
- [文件结构](#文件结构)
- [编译gfs](#编译gfs)
  * [老方法](#老方法)
  * [cmake方法](#cmake方法)
- [参考](#参考)

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

#### 普通环境

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

#### docker环境

该镜像已经构建好 拉取命令 `docker pull opeceipeno/dgl:devel-gpu-lite`

##### Dockerfile.devel-gpu

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

##### Dockerfile.devel-gpu-compiled

该镜像是在上述镜像的基础上，提前编译好了`dgl0.1.x`和`0.7.x`，已在阿里云机器上

```dockerfile
FROM opeceipeno/dgl:devel-gpu-lite

# 0.1.x
SHELL ["conda", "run", "-n", "dgl-0.1.x", "/bin/bash", "-c"]

RUN mkdir -p /workspace/0.1.x/build && \
    cd /workspace/0.1.x/build && \
    cmake .. && \
    make -j4 && \
    cd /workspace/0.1.x/python && \
    python setup.py install

# 0.7.x
SHELL ["conda", "run", "-n", "dgl-0.7.x", "/bin/bash", "-c"]

RUN mkdir -p /workspace/0.7.x/build && \
    cd /workspace/0.7.x/build && \
    cmake -DUSE_CUDA=ON -DBUILD_TORCH=ON .. && \
    make -j4 && \
    cd /workspace/0.7.x/python && \
    python setup.py install

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
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

### 编译gfs

#### 老方法

下载依赖库：

```
wget https://github.com/fmtlib/fmt/archive/refs/tags/4.1.0.zip
wget https://github.com/sparsehash/sparsehash/archive/refs/tags/sparsehash-2.0.4.zip
```

编译fmt-4.1.0

```
mkdir build && cd build
cmake ..
make -j 8
sudo make install
```

编译sparsehash-sparsehash-2.0.4

```
./configure
make -j 8
sudo make install
```

正式编译gfs：

```
cd gfs/env && make
cd gfs/fs && make
cd gfs/metrics && make
cd monitoring && make
cd util && make
cd util/threadpool && make
```

期间可能有报错如下：

```
# 缺少：zlib-devel
# dgl/gfs/fs/../../gfs/util/ioutil.h:36:18: 致命错误：zlib.h：没有那个文件或目录
# 则安装zlib-devel
yum install zlib-devel
```

编译util/threadpool为空时，可以先直接用c++编译：

```
g++ -g -std=c++11 -Wall -DSKG_EDGE_DATA_COLUMN_STOAGE -DSKG_PROPERTIES_SUPPORT_NULL -D_FILE_OFFSET_BITS=64 -DDB_ADAPTER -DSKG_DISABLE_COMPRESSION -DSKG_PREPROCESS_DYNAMIC_EDGE -DUSE_STL_PRIORITY_QUEUE -DSKG_SUPPORT_THREAD_LOCAL -DSKG_QUERY_USE_MT -DSKG_REQ_VAR_PROP -I/data/jiangjiajun750/cpp/dgl/gfs/util/threadpool/../../../gfs -c thread_pool_impl.cc -o /data/jiangjiajun750/cpp/dgl/gfs/util/../../gfs/obj/thread_pool_impl.o
```

构建lib和测试：

```
cp fmt-4.1.0/build/fmt/libfmt.a gfs/test/
cd gfs/test && make lib
cd gfs/test && make newg
./newg
```

#### cmake方法

##### cpp编译和调试

目前已将gfs整体编译至dgl的so中，使用cmake统一管理编译，具体可查看：`CMakeLists.txt 和gfs/CMakeLists.txt`。下一步计划打通python调用。

下面是使用cmake编译gfs+dgl的具体步骤。

下载和更新代码仓库：

```
# 分支为0.1.x_dev
git clone https://github.com/Qksidmx/dgl

# 更新子模块
git submodule update --init --recursive
```

编译安装fmt-4.1.0到系统中（已安装可忽略）

```
cd dgl/gfs/third_party/fmt && mkdir build && cd build
cmake ..
make -j 8
sudo make install
```

编译安装sparsehash-2.0.4到系统中（已安装可忽略）

```
cd dgl/gfs/third_party/sparsehash
./configure
make -j 8
sudo make install
```

正式编译gfs+dgl：

```
mkdir build && cd build
cmake .. && make -j 8
# 测试：
./newg
```

其中`libdgl.so` 和 `newg` 将生成在build目录下，可以直接使用。编译好一次以后，再改动c++代码的话，可以简化执行指令和加速编译：

```
cd build && make -j 8 && cp libdgl.so ../lib && cp libdgl.so ../python/build/lib
```

##### python编译和调试

进入python虚拟环境，conda、virtualenv或者docker均可

```
# virtualenv ，环境初始化参考前面部分
source ./venv/bin/activate
```

编译python

```
cd python && python3 setup.py build
```

执行python测试脚本：

```
cd tests/graph_index && python3 test_skg_graph.py
```

### 参考

* openmp多线程编程：https://blog.csdn.net/acaiwlj/article/details/49818965
* makefile处理头文件依赖关系：https://blog.csdn.net/dlf1769/article/details/78997967
* 合并静态库的最佳实践：https://zhuanlan.zhihu.com/p/389448385
* cmake添加自定义操作：https://zhuanlan.zhihu.com/p/95771200
