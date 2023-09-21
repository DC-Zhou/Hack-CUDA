## 前言
本文描述以下知识点：

- [ ] 基于cifar-10的模型训练，
- [ ] onnx模型导出
- [ ] 通过trtexec转fp16和int8模型，并使用Nsight分析转换过程；
- [ ] 通过Python转换fp16和int8模型
- [ ] 对int8模型使用数据进行校准（Calibration）

今天来整理之前复现jetson上的DLA Core使用入门材料：[jetson_dla_tutorial](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial)。

DLA全称Deep Learning Accelerator简单来说是Nvidia在嵌入式设备上实现的用于加速神经网络层的硬件核心，类似RKNN开发板上的NPU。算是继tensor core以后目前最新的加速core（当然Nvidia卡上加速神经网络的东西很多）。由于DLA core是物理核心，目前支持DLA core的设备就不像tensor core那么多了，可以参看下表：

| 设备   | DLAv1.0 | DLAv2.0 |
| ----- | -----    | ----- | ----- |
| jetson xavier nx series | 支持 | x |
| jetson orin series | x | 支持 |
| NVIDIA DRIVE Hyperion | x | 支持 |
Nvidia凭借做硬件的技术优势，逐步在降维打击国内外的中小算例嵌入式开发板厂商。Jetson Orin系列基本上就是老黄一统江湖的野心体现。Jetson基本上是单位瓦数里能达到的最高算力（当然价格基本也是最高）。在用于SLAM，自动驾驶方面都有潜力。[jetson_dla_tutorial](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial)这篇教程相对简单，包括以下几方面：

* **torch模型转换tensorRT模型**
* **Nsight System 做性能分析**
* **DLA core 基础使用**

作为jetson的在深度学习方面应用的入门材料很不错，进而编写了这篇复现的文章，希望有错误的地方，读者不吝惜赐教。
## 准备环境和数据集
先说说硬件环境，我使用的是一块64GB的jetson Agx Orin toolkit，算是官方推荐设备。官方教程推荐使用jetson Orin系列（jetson Agx Orin和Jetson Orin nano），我估计NVIDIA DRIVE Hyperion也是可以的。
git clone项目：
```shell
git clone https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial.git
# home/jou/Documents/jetson_dla_tutorial
mdkir data/cifar
```
下载cifar-10的数据集压缩包，放到data/cifar目录下（也可以自己用迅雷之类的下载，放到同一目录即可）
```shell
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
cp cifar-10-binary.tar.gz /home/jou/Documents/jetson_dla_tutorial/data/cifar
```
这里将使用docker管理环境，docker镜像可以从[NGC](https://catalog.ngc.nvidia.com/containers)查询 和拉取(**Docker镜像需要和你使用Jetson的Jetpack对齐**）：
```shell
sudo docker pull nvcr.io/nvidia/l4t-ml:r35.2.1-py3
```
启动docker时， 将项目挂载进docker
```shell
sudo docker run -it --rm --runtime nvidia --network host -v /home/jou/Documents/jetson_dla_tutorial:/home/ nvcr.io/nvidia/l4t-ml:r35.2.1-py3
```
这样基础的工作就准备好了。接下来，让我们看看[jetson_dla_tutorial](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial)

## 训练model_GN模型
docker中包含所有的pip包，所以我们不需要重下pip包。
model_gn的定义在models.py 中，结构大致如下：
```python
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, 10)
```
训练该模型则使用train.py文件，指令如下：
```shell
python3 train.py model_gn --checkpoint_path=data/model_gn.pth
```
执行train.py以后，程序会自动检测是否在data目录下包含cifar-10的数据集，所以之前要把数据集保存在这个目录下。50epoch训练时间在30分钟左右，本项目主要走为了流程，所以可以在train.py中把训练epoch改成10，以节省时间。
最后我们需要定义导出成onnx的数据结构，如下：
``` python
data = torch.zeros(1, 3, 32, 32).cuda()

torch.onnx.export(model_gn, data, 'model_gn.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```
使用pth文件导出到onnx模型：
```shell
python3 export.py model_gn data/model_gn.onnx --checkpoint_path=data/model_gn.pth
```
实际业务中由于模型结构复杂，可能还需要做onnx-simplfy。这里模型比较简单， 就跳过这步了。

## model_GN模型转trt模型
将onnx模型转换成trt模型本质上是通过trt来读取onnx模型后重构（build）成trt模型的过程。主要方法有两种：trtexec转换工具和tensorrt Python接口，下面我将先展示trtexec接口的使用：
``` shell
alias trtexec=/usr/src/tensorrt/bin/trtexec
trtexec --onnx=model_gn.onnx --shapes=input:32x3x32x32 --saveEngine=model_gn.engine --exportProfile=model_gn.json --int8 --useDLACore=0 --allowGPUFallback --useSpinWait --separateProfileRun > model_gn.log
```
主要参数解释如下：
- --onnx：读取onnx模型
- --shapes：设定输入模型的tensor大小（这里名称要与netron.app打开的onnx名称对应）
- --saveEngine：保存trt模型位置
- --exportProfile：使用随机参数做profle 保存profile结果到
- --int8：量化成int8模型
- --useDLACore = 0 ：使用DLA core
- --allGPUFallback ：允许将DLA不支持的layer转到tensort中处理
- --useSpinWait：让CPU主动去做GPU context 切换
- --separateProfileRun 启用基准测试中的性能分析

在上述的CLI中，trtexec为导出的model_gn使用随机数填充做了数据基准测试，并且把测试结果保存在model_gn.json中，我们可以分析model_gn.json，其中有两部分比较重要：
1.对于该模型GPU layer 对比DLA layer分配情况：
![](/img/bVc9zuK)
2.对于该模型使用随机数推理性能分析：
![](/img/bVc9zuL)
以上两个情况说明：

1. 当前模型发生DLA Layer 到GPU layer 的context swicth比较多；
2. model_gn模型的性能为357.164qps

但是以上数据的分析都很粗略，我们需要更加详细的信息来下一步的优化，比如每层的执行时间，总共发生多少次context switch等，于是，我们需要引入Nsight system来分析整个过程。
s
## Msight System分析model_GN模型
Nvidia上的分析工具有很多，从Nvpovf 到MemoryCheck。自2013年开始，Nvidia开始推荐在Nsight下的三个主要分析工具：
- Nsight Systems:适用于系统级分析 （Nvidia Nvprof也包含在其中）
- Nsight Compute：适用于Kernel分析
- Nsight Graphic：适用于图像性能分析

下面我们通过CLI启动Nsight Systems中的nvprofile工具对trtexec的随机数据测试性能过程做分析

```shell
sudo nsys profile --trace=cuda,nvtx,cublas,cudla,cusparse,cudnn,nvmedia --output=model_gn.nvvp /usr/src/tensorrt/bin/trtexec --loadEngine=model_gn.engine --iterations=10 --idleTime=500 --duration=0 --useSpinWait
```
主要参数解释如下：
- --trace==cuda,nvtx,cublas,cudla,cusparse,cudnn,nvmedia：定义在调用过程中捕获哪些事件，cudla用来捕获dla使用，nvtx用来捕获tensorrt layer的切换；
- --output=model_gn.nvvp：输出数据结果；
- --iterations=10 ：迭代次数
- --idleTime=500：在迭代之间插入500ms的堵塞，以便我们更好分辨
- --useSpinWait：在context switch之间加入显式的CPU同步

如果你使用sudo模式，将在当前目录下生成分析报告，如果你没使用sudo模式，分析报告将在/tmp/文件夹下生成：
``` shell
Generating '/tmp/nsys-report-259c.qdstrm'
[1/1] [========================100%] model_gn.nvvp.nsys-rep
Generated:
    /home/jou/Documents/jetson_dla_tutorial/data/model_gn.nvvp.nsys-rep
```
可以将这个分析报告拷贝到任何机器上的Nsight Systems打开，你能看到如下图：
![](F:\Hack-CUDA\jetson_dla_tutorial\img\model_gn_nsys_nvprof.png)
这个报告粗略有两个部分，左边的蓝笔部分包含的是trtexec对trt模型(engine）的解析过程，右侧部分是使用随机数据填充，对模型进行的10次迭代测试（这也体现冷启动的第一次总是会比其余几次慢一点)。我们选一次迭代的数据放大如下图：
![](F:\Hack-CUDA\jetson_dla_tutorial\img\model_gn_per.JPG)
其中灰色的部分使用cuDLA API，黄色使用tensorRT API，我们可以发现，每次处理GroupNrom的时候，就会发生cuDLA context 和tensorRT context switch。这个模型中发生多次交换，数据就需要重新传输，这种是比较影响性能的点；
接下来我们尝试优化一下。

## model_GN的改进模型model_BN模型
我们把model_gn中的所有GroupNorm 换成BathNorm，就能得到model_bn模型：
```python
class ModelBN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
```
由于这个模型输入和输出结构没有改变，我们依旧使用model_gn的方法导出和训练：
训练：
```shell
python3 train.py model_gn --checkpoint_path=data/model_gn.pth
```
导出onnx模型：
```shell
python3 export.py model_bn data/model_bn.onnx --checkpoint_path=data/model_bn.pth
```
将model导出成trt模型：
```shell
trtexec --onnx=model_bn.onnx --shapes=input:32x3x32x32 --saveEngine=model_bn.engine --exportProfile=model_bn.json --int8 --useDLACore=0 --allowGPUFallback --useSpinWait --separateProfileRun > model_bn.log
```
看看model_bn.log:
```shell
[08/31/2023-07:07:49] [I] [TRT] ---------- Layers Running on DLA ----------
[08/31/2023-07:07:49] [I] [TRT] [DlaLayer] {ForeignNode[/cnn/cnn.0/Conv.../cnn/cnn.11/Relu]}
[08/31/2023-07:07:49] [I] [TRT] [DlaLayer] {ForeignNode[/linear/Gemm]}
[08/31/2023-07:07:49] [I] [TRT] ---------- Layers Running on GPU ----------
[08/31/2023-07:07:49] [I] [TRT] [GpuLayer] POOLING: /pool/GlobalAveragePool
[08/31/2023-07:07:49] [I] [TRT] [GpuLayer] SHUFFLE: reshape_after_/linear/Gemm
...
[08/31/2023-07:07:55] [I] 
[08/31/2023-07:07:55] [I] === Performance summary ===
[08/31/2023-07:07:55] [I] Throughput: 1663.19 qps
[08/31/2023-07:07:55] [I] Latency: min = 0.617432 ms, max = 0.829102 ms, mean = 0.634685 ms, median = 0.629639 ms, percentile(90%) = 0.657227 ms, percentile(95%) = 0.675781 ms, percentile(99%) = 0.689697 ms
```
现在这个样子比model_gn.log看起来好多了，我们再去看看单次迭代中context情况，分析模型：
```shell
nsys profile --trace=cuda,nvtx,cublas,cudla,cusparse,cudnn,nvmedia --output=model_bn.nvvp /usr/src/tensorrt/bin/trtexec --loadEngine=model_bn.engine --iterations=10 --idleTime=500 --duration=0 --useSpinWait
```
生成图像：

![](F:\Hack-CUDA\jetson_dla_tutorial\img\BN_Nsight.webp)

我们可以看到，调整模型结构，使得cuDLA和tensorRT之间的数据交换减少了不少。减少了context switch的事件，进而减少总耗时，提高了计算性能。

## 使用TensorRT Python API导出数据
接下来我们使用TensorAPI导出trt模型，并且使用原始数据集来校准我们的trt模型。可以参考下面代码片段。
1.使用tensorrt读取onnx模型
```python
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
builder.max_batch_size = args.batch_size
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(args.onnx, 'rb') as f:
    parser.parse(f.read())
```
2.启用profile 和定义profile config
```python
profile = builder.create_optimization_profile()
profile.set_shape(
    'input',
    (args.batch_size, 3, 32, 32),
    (args.batch_size, 3, 32, 32),
    (args.batch_size, 3, 32, 32)
)
```
3.为trt模型启用DLA，设定build config
``` python
if args.int8:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = DatasetCalibrator(data, train_dataset)

if args.dla_core is not None:
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = args.dla_core

if args.gpu_fallback:
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    
config.add_optimization_profile(profile)
config.set_calibration_profile(profile)

engine = builder.build_serialized_network(network, config)
```
## 数据集校准（Calibration）模型
因为我们已经有完整的cifar-10数据集，为了追求model_bn.engine的精度，我们还可以在用数据集对导出的模型进行校准，具体可以参看如下
先设计calibrator.py用于生成给DLA int8模型的数据集：
```python
import torch
import tensorrt as trt

__all__ = [
    'DatasetCalibrator'
]

class DatasetCalibrator(trt.IInt8Calibrator):
    
    def __init__(self, 
            input, dataset, 
            algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        super(DatasetCalibrator, self).__init__()
        self.dataset = dataset
        self.algorithm = algorithm
        self.buffer = torch.zeros_like(input).contiguous()
        self.count = 0
        
    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):
            for buffer_idx in range(self.get_batch_size()):
                dataset_idx = self.count % len(self.dataset) # roll around if not multiple of dataset
                image, _ = self.dataset[dataset_idx]
                image = image.to(self.buffer.device)
                self.buffer[buffer_idx].copy_(image)

                self.count += 1
            return [int(self.buffer.data_ptr())]
        else:
            return []
        
    def get_algorithm(self):
        return self.algorithm
    
    def get_batch_size(self):
        return int(self.buffer.shape[0])
    
    def read_calibration_cache(self, *args, **kwargs):
        return None
    
    def write_calibration_cache(self, cache, *args, **kwargs):
        pass
```
使用该数据集来校准DLA int8模型：
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root=os.path.join(args.data_dir, 'cifar10'), 
    train=True,
    download=True, 
    transform=transform
)

batch_size = 32

data = torch.zeros(batch_size, 3, 32, 32).cuda()

config.int8_calibrator = DatasetCalibrator(data, train_dataset)
```
最后我们使用build.py来输出engine模型：
```shell
python3 build.py data/model_bn.onnx --output=data/model_bn.engine --int8 --dla_core=0 --gpu_fallback --batch_size=32
```
## 评估（eval）模型的准确性
需要知道这个模型在新的数据上的效果，我们需要评估这个模型，评估过程分为以下三步：
1.通过torch创建数据集；
2.通过tensorrt的python API读取模型，创建tensorrt的context；
3.配置tensorrt的runtime的输入和输出；
具体操作细节可以参考eval.py，这里由于篇幅关系就不详述了。
## 总结
这篇文章复现基本没有难度，库基本都在docker里，算是入门级别的比较好的文章。
## 补充点
1.GPU上执行程序，都会创建context来实现对资源的管理和分配，以及异常处理等情况。event 和stream 更加细分的归属于context的概念。而handle代表一个更加具体的指代（类似于内存空间和指针）；
2.docker基于快照理念，docker image在退出后会恢复到快照，除非对docker images做commit，否则所有修改都会丢失（修改类似于git add）；
3.DLA core本身的使用有两种模式：
- 混合模型：与trt混用，config中允许GPU_CallBack;
- 独立模型：模型全部搭载进DLA core，config不允许GPU_CallBack；

本文主要介绍混合模型，至于独立模型的方式，可以参考接下来的关于cuDLA-sample的文章。