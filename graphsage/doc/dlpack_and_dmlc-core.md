<center>dlpack和dmlc-core</center>

### dlpack

dlpack不会实现tensor和op，而是能让不同深度训练框架之间复用tensor和op
dlpack的api很简单，然我们可以放弃关注硬件接口，分配器，专注于最小的数据结构，能支持跨设备的数据存储和展示

枚举：

```
DLDeviceType
DLDataTypeCode
```

Struct：

```
DLDevice
DLDataType
DLTensor
DLManagedTensor
```

例子：

```python
from torch.utils.dlpack import to_dlpack
t = torch.ones((5,5), dtype=torch.float32, device=torch.device('cpu'))
dlp = to_dlpack(t)

from nnabla.utils.dlpack import from_dlpack
arr = from_dlpack(dlp)
```



### ctypes

python可利用ctypes机制来调用c/c++模块，对基本类型的映射有良好的支持，如下举例子：

g++ -fPIC -shared great_module.c -o great_module.so

python前加星号，用于将列表或字典等参数拆解成单个参数


### dmlc-core

是dmlc的基础库，文档较少，文档只有param.h，即解析参数的库

可以通过引用头文件的形式来利用这些库

直接翻源码，有几个主目录：

```
include 基础库
src/data 解析数据的迭代器
src/io 处理数据和文件流的io库
test 单元测试目录，因文档欠缺，可以通过这个目录熟悉基础库基本用法
tracker 提交任务的python脚本，可以提交任务至k8s/yarn/mesos/ssh 等
```

include文件夹下的文件列举如下：

```
any.h 可以包装容纳任意数据结构的库
array_view.h 用于指向array的只读的数据结构
blockingconcurrentqueue.h 高效的阻塞并发队列，使用信号量（semaphore）实现
concurrency.h 提供线程安全的数据结构
concurrentqueue.h 高效的多生产者、多消费者的无锁队列
filesystem.h 操作文件的库
input_split_shuffle.h 随机拆分输入数据的库
io.h 定义数据结构的序列化接口
json.h 轻量级的json读写库用于c++数据结构，支持stl组件/结构
logging.h 日志库，使用glog实现，若无glog，使用自己的底层实现
lua.h 用于和lua与torch交互的头文件，作者有陈天奇
memory_io.h 用于序列化数据结构至内存中
optional.h 用于存放optional类数据的容器
parameter.h 参数的设置和检查库
recordio.h 打包二进制数据为拆分的格式，方便在二进制下交换数据，例如纯二进制数据和Protobuf
strtonum.h str to float和str to double的高效实现
threadediter.h 基于线程的迭代器，实现通用的基于线程的流水线例如预获取或者预计算；作者是陈天奇
thread_group.h 提供线程池功能，有原始的同步，以及生命周期管理
thread_local.h 可插拔的线程本地存储
timer.h 计时器；作者是陈天奇
```

src/data 目录文件枚举如下：

```
basic_row_iter.h 用于加载数据到内存的基于行的迭代器
csv_parser.h 以迭代的形式解析csv文件
disk_row_iter.h 基于行的迭代器，用于缓存数据到磁盘
libfm_parser.h 解析libfm格式的迭代器
libsvm_parser.h 解析libsvm格式的迭代器
parquet_parser.h 解析parquet格式的迭代器
row_block.h 用于支持row block数据结构
text_parser.h 解析文本格式的迭代器
```

src/io 目录文件枚举如下：

```
azure_filesys.cc 读写azure云上的文件
cached_input_split.h 用于缓存数据至磁盘，并后续从磁盘读取
filesys.cc 提供便利文件夹和删除文件夹的方法
hdfs_filesys.cc 读写hdfs上的文件
indexed_recordio_split.cc 用于拆分已索引的recordio文件
input_split_base.h 从多个文件来拆分输入流的基类
line_split.h 基于行的从多个文件拆分输入流
local_filesys.cc 读写本地的文件
recordio_split.h 基于recordio文件的从多个文件拆分输入流
s3_filesys.cc 读写亚马逊云上的文件
single_file_split.h TODO
single_threaded_input_split.h 用于调试目的
threaded_input_split.h InputSplit的多线程版本，有一个prefetch的线程会先进行预获取
uri_spec.h TODO
```


