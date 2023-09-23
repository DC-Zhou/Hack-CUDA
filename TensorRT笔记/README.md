# TensorRT 笔记一：基础知识

## 前言

TensorRT是由NVIDIA开发的高性能深度学习推理引擎。它针对NVIDIA GPU进行了优化，可以加速深度神经网络（DNN）的推理过程，提供低延迟和高吞吐量。TensorRT使用精确的数值计算和内存优化技术，以及自动融合和裁剪网络层等技术，实现了高效的推理。

TensorRT支持常见的深度学习框架，如TensorFlow、PyTorch和Caffe等，并提供了可扩展的API，使用户能够轻松地将其现有的训练模型转换为TensorRT可用的格式。通过优化网络结构和减少冗余操作，TensorRT能够提高推理速度，并在保持模型准确性的同时降低内存消耗。

除了基本的推理功能，TensorRT还提供了一些高级特性。其中包括动态形状推理，允许在运行时处理具有不同输入形状的模型；INT8量化，通过降低计算精度来进一步提高推理性能；以及多个GPU的并行推理，以加速大规模模型的推理过程。

TensorRT已经被广泛应用于各种领域，包括自动驾驶、机器人技术、医疗影像处理和自然语言处理等。其高性能和灵活性使得TensorRT成为深度学习模型在实际应用中的理想选择，能够帮助用户更高效地部署和推理他们的模型。

本文中，将为您介绍转换TensorRT模型的两种种不同的方法；

请注意，在使用下列方法之前，需要您确认本地的CUDA版本，并且在[此处](https://developer.nvidia.com/tensorrt-download)下载TensorRT的官方包，以下方法均基于TensorRT官方包实现；

对于Windows平台还需要下载CuDNN包，请在[此处]()下载

TensorRT模型命名遵循以下原则，由于trt模型本身是二进制模型，故支持trt和engine作文件类型，二者使用上并无差别：

- 模块名+精度 ： 比如：tumble.fp16.engine
- QAT和PTQ模型需要标记使用：比如：tumble_qat.engine
- 使用特殊的half和int8的IOFormats的模型需要标明： tumble.fp16infp16out.engine 

## 通过转换工具转换

下载解压出的TensorRT包在bin目录下自带转换工具trtexec.exe

### 转换fp16模型

#### onnx模型转换TensorRT模型T

```shell
trtexec --onnx=<onnx_file> --fp16 --saveEngine=<engine_file>
```

#### caffe模型转换TensorRT模型

```shell
trtexec --deploy=<caffe_prototxt_file> --model=<caffe_model_file> --output=<output_node_name> --workspace=<workspace_size> --fp16 --saveEngine=<engine_file>
```

对于模型输出不固定的模型，需要增加--shape参数

```shell
trtexec --shapes=input:32x3x32x32 # 模型input参数+shape
```

### 转换int8模型：

int8模型需要对数据进行校准，所以需要额外提供用于量化的数据

#### onnx模型转换TensorRT模型

```shell
trtexec --onnx=tumble.onnx --explicitBatch --calibrationData=<CALIBRATION_DATA_PATH> --saveEngine=tumble_int8.engine --int8
```

#### caffe模型转换TensorRT模型

```shell
trtexec --deploy=<caffe_prototxt_file> --model=<caffe_model_file> --explicitBatch --calibrationData=<CALIBRATION_DATA_PATH> --saveEngine=tumble_int8.engine --int8
```

对于使用的<CALIBRATION_DATA_PATH>结构需要遵循以下文件结构：

```shell
----<CALIBRATION_DATA_PATH>：
	---- class1
		--- 0.jpg
		--- 1.jpg
	---- class2
		--- 0.jpg
		--- 1.jpg
```

### QAT量化和PTQ量化：

TensorRT是NVIDIA开发的深度学习推理优化器，旨在加速和优化神经网络模型的推理过程。为了进一步提高性能并减少模型的存储需求，TensorRT引入了两种量化模式，`implicitly`以及`explicitly`量化技术，前者是隐式量化技术，在TensorRT7版本用的比较多，后者就是我们讨论的显式量化技术，在TensorRT8版本开始得到普遍使用。

**QAT（Quantization-Aware Training）量化技术**是一种在训练过程中考虑量化误差的方法。传统上，在将浮点模型量化为低精度（如8位整数）时，会导致精度损失。为了解决这个问题，QAT在训练期间通过模拟量化过程来估计量化误差，并将其纳入损失函数中进行优化。这样可以使得模型在量化后仍然具有较高的准确性。

使用QAT量化技术时，开发者需要将模型定义为支持动态范围量化的方式，并在训练期间使用特定的量化操作来模拟推理中的量化过程。这样训练出来的模型就能够更好地适应量化带来的精度损失，从而在推理阶段获得更好的性能。

**PTQ（Post-Training Quantization）量化技术**是一种在训练后对已有的浮点模型进行量化的方法。相较于QAT，PTQ不需要在训练期间考虑量化误差，而是直接对已经训练好的模型进行量化操作。这种技术通过将模型中的权重和激活值从浮点数表示转换为低精度的整数表示来减少计算资源的使用。

在TensorRT中，PTQ可以应用于已经训练好的模型，并且可以选择不同比特宽度的量化，如8位或16位等。PTQ技术通过使用离线校准数据集来确定动态范围，并将动态范围量化为离散值。然后，量化后的模型可以在推理过程中以更低的计算资源需求运行，提高推理性能。

总的来说，QAT和PTQ是TensorRT中两种常用的量化技术。QAT通过在训练期间考虑量化误差来优化模型，而PTQ则是在训练后对已有模型进行离散量化，从而减少计算资源需求。这些技术可以帮助在保持模型性能的同时提高推理效率。

进一步的详细情况可以参考[本文](https://zhuanlan.zhihu.com/p/479101029)

对于trtexec接口是不支持使用QAT和PTQ量化，我们可以使用Python接口结合TensorRT的相关库来实现以上技术。

### 使用Half和int8的IOFormat的模型：

一般来说，当你转换出fp16或者int8模型时，模型中的权重将从原f32数据量化到对应的精度。但是通常用输入的为tensor为f32精度。因此，在使用TensorRT模型推理过程中必然发生f32到f16 或者f32到int8的数据类型转换；

数据类型转换有两种方式（以fp16为例）：

- trt-API转换；
- CUDA_fp16库转换：

trt-API转换来自TensorRT-API中，当你使用模型转换成trt模型，本质上是对原模型的结构和使用trt-api重新Build的过程，因此就可以在trt-api中插入转换层，以实现f32数据转fp16数据的效果；

CUDA_fp16库转换则是外置的转换过程，需要调用指令将float数据转换成对应的fp16数据，然后再输入模型中的过程，这个过程受外界控制，适合对大规模的数据进行处理；

而当我们使用trtexec的过程中，也可以显式指定TensorRT模型的IOFormat，以实现模型上的精简：

```shell
trtexec --onnx=<onnx_path> --verbose --fp16 -- saveEngin<onnx.fp16.fp16chw16in.fp16chw16out.engine> --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16
```

对于int8

```
trtexc --shapes=images:1x3x672x672 --onnx=<onnx_path> --useDLACore=0 --buildDLAStandalone --saveEngine=<onnx.engine)> --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8
```

使用CUDA_fp16进行模型精度转化可以参考[该项目](https://github.com/NVIDIA-AI-IOT/cuDLA-samples/tree/main)

### DLA core的调用

Nvidia的DLA（Deep Learning Accelerator）核心是一种专门用于深度学习推理的硬件加速器。它被集成在Nvidia Jetson系列嵌入式平台和Nvidia驱动设备中，旨在提供高效的推理性能。

在Nvidia Jetson和Nvidia DRIVE平台上，DLA核心被用作Tensor Core的补充，以加速神经网络的推理任务。通过利用DLA核心，Jetson平台可以在边缘设备上进行快速、低功耗的深度学习推理。

在TensorRT中，DLA core的使用有两种方法：

- 混合模式
- 独立模式

**混合模式**下，DLA core受到TensorRT Engine中的调度管理，可以显式指定trt是否运行在DLA core上，是否允许Layer在DLA core 和 GPU上的切换：

```shell
trtexec --onnx=<onnx_path>  --saveEngine=onnx.engine --exportProfile=onnx.json --int8 --useDLACore=0 --allowGPUFallback --useDLACore=0
```

**独立模式**下，DLA通过cudla_context 接口以实现对DLA core的调用。同时在模型转换过程中，需要设定TensorRT转换出的模型支持dlastandalone功能

```shell
onnx --onnx=<onnx_path> --fp16 --saveEngine=onnx.standalone.trt  --buildDLAStandalone --useDLACore=0
```

### Int8模型部分节点使用fp16精度

对于部分Int8支持不够好的模型，可以针对某些层，使用fp16精度：

```shell
trtexc --shapes=images:1x3x672x672 --onnx=<onnx_path> --useDLACore=0 --buildDLAStandalone --saveEngine=<onnx.engine> --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 --calib=data/model/qat2ptq.cache --precisionConstraints=obey --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16,"/model.23/cv3/conv/Conv":fp16,"/model.23/cv3/act/Sigmoid":fp16,"/model.23/cv3/act/Mul":fp16
```

### 使用随机数调优模型性能

在trtexec转换过程中使用如下显式标识符，可以显式使用 --useSpinWait --separateProfileRun来指定随机数填充测试下的模型推理性能：

```shell
trtexec --onnx=<onnx_path>--shapes=input:32x3x32x32 --saveEngine=onnx.engine --exportProfile=model_gn.json --int8 --allowGPUFallback --useSpinWait --separateProfileRun > onnx.log
```

可以得到如下图类似的模型运行性能提示：

```shell
[09/06/2023-10:10:26] [I] === Trace details ===
[09/06/2023-10:10:26] [I] Trace averages of 10 runs:
[09/06/2023-10:10:26] [I] Average on 10 runs - GPU latency: 70.9993 ms - Host latency: 72.3481 ms (enqueue 0.269727 ms)
[09/06/2023-10:10:26] [I] Average on 10 runs - GPU latency: 70.9871 ms - Host latency: 72.3341 ms (enqueue 0.245013 ms)
[09/06/2023-10:10:26] [I] Average on 10 runs - GPU latency: 70.9935 ms - Host latency: 72.3413 ms (enqueue 0.286682 ms)
[09/06/2023-10:10:26] [I] Average on 10 runs - GPU latency: 71.0039 ms - Host latency: 72.3516 ms (enqueue 0.261987 ms)
[09/06/2023-10:10:26] [I] 
[09/06/2023-10:10:26] [I] === Performance summary ===
[09/06/2023-10:10:26] [I] Throughput: 13.7793 qps
[09/06/2023-10:10:26] [I] Latency: min = 72.2924 ms, max = 72.4907 ms, mean = 72.3422 ms, median = 72.3247 ms, percentile(90%) = 72.3965 ms, percentile(95%) = 72.4125 ms, percentile(99%) = 72.4907 ms
[09/06/2023-10:10:26] [I] Enqueue Time: min = 0.233765 ms, max = 0.538818 ms, mean = 0.26318 ms, median = 0.252014 ms, percentile(90%) = 0.295166 ms, percentile(95%) = 0.307983 ms, percentile(99%) = 0.538818 ms
[09/06/2023-10:10:26] [I] H2D Latency: min = 0.484131 ms, max = 0.492859 ms, mean = 0.485587 ms, median = 0.485291 ms, percentile(90%) = 0.486572 ms, percentile(95%) = 0.488831 ms, percentile(99%) = 0.492859 ms
[09/06/2023-10:10:26] [I] GPU Compute Time: min = 70.946 ms, max = 71.1436 ms, mean = 70.9949 ms, median = 70.9772 ms, percentile(90%) = 71.0479 ms, percentile(95%) = 71.0645 ms, percentile(99%) = 71.1436 ms
[09/06/2023-10:10:26] [I] D2H Latency: min = 0.84375 ms, max = 0.864258 ms, mean = 0.861699 ms, median = 0.862122 ms, percentile(90%) = 0.862549 ms, percentile(95%) = 0.86319 ms, percentile(99%) = 0.864258 ms
[09/06/2023-10:10:26] [I] Total Host Walltime: 3.26577 s
[09/06/2023-10:10:26] [I] Total GPU Compute Time: 3.19477 s
[09/06/2023-10:10:26] [I] Explanations of the performance metrics are printed in the verbose logs.
```

## 通过Python接口转换

通过Python接口转换主要描述转换fp16和int8模型，其他功能暂不描述；

使用Python接口转换TensorRT模型主要分为以下三个部分：

- 构建Python-tensorRT环境

- 使用export.py转换pt模型为onnx模型
- 使用builder.py构建tensorRT模型
- 构建QAT和PTQ模型

### Python-TensorRT环境

安装tensorRT环境

```shell
pip Install tensorrt
# tensorRT安装可能会比较慢， 因为其会自动安装cuda-toolkit 建议在虚拟环境下使用
# 安装Nvidia官方量化和图优化工具
nvidia-pyindex>=1.0.9
onnx_graphsurgeon>=0.3.11
```

### 使用export.py转换pt模型为onnx模型

在导出之前，需要在export.py中设置好模型的输入和输出：

```python
data = torch.zeros(1, 3, 32, 32).cuda()

torch.onnx.export(
    model,
    data,
    args.output,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

然后使用以下指令导出：

```shell
python3 export.py yolov5 data/yolov5.onnx --checkpoint_path=data/yolov5.pt
```

### 使用builder.py构建tensorRT模型

开始构建之前，需要到builder.py文件中确定归一化参数和profile size

```Python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
...
profile = builder.create_optimization_profile()
profile.set_shape(
    'input',
    (args.batch_size, 3, 32, 32),
    (args.batch_size, 3, 32, 32),
    (args.batch_size, 3, 32, 32)
)
```

然后按照以下指令导出：

fp16模型导出

```shell
python3 build.py data/onnx.onnx --output=data/onnx.engine --fp16 --batch_size=32
```

int8模型导出

```shell
python3 build.py data/onnx.onnx --output=data/onnx.engine --int8 --data_path=<data_path> --batch_size=32
```

目前所有单图片推理模型batch_size可以全部设置为1；

### 构建QAT和PTQ模型（待更新）







