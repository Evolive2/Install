# 从Cuda到Pytorch(GPU)的安装
# 0. Need To Know
通常，我们首先在conda上安装虚拟环境，设定了python的版本；这不是一种好的流程，它会导致你在之后的torch和cuda的选择中，迫使你更换python版本。
事实上，我们认为在确定所有版本互相匹配后，再进行环境创建是保险的，从cuda>torch>python的选择是合理的。
# 1. 选择Cuda
第一步，必须了解所用GPU的算力。
通过命令行键入：
```
nvidia-smi
```
输出：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.57       Driver Version: 515.57       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A6000    Off  | 00000000:51:00.0 Off |                  Off |
| 48%   75C    P2   204W / 300W |  19158MiB / 49140MiB |     65%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A6000    Off  | 00000000:C3:00.0 Off |                  Off |
| 46%   73C    P2   195W / 300W |  12504MiB / 49140MiB |     73%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1139      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A     11670      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A    131110      C   ...-3.8-1.10-11.3/bin/python    19145MiB |
|    1   N/A  N/A      1139      G   /usr/lib/xorg/Xorg                  4MiB |
|    1   N/A  N/A     11670      G   /usr/lib/xorg/Xorg                  4MiB |
|    1   N/A  N/A    131111      C   ...-3.8-1.10-11.3/bin/python    12491MiB |
+-----------------------------------------------------------------------------+
```
可以看到我的GPU为两块NVIDIA RTX A6000，驱动版本为Driver Version: 515.57，最高支持CUDA Version: 11.7（并非表示已安装成功版本）
然后，查找GPU算力，https://developer.nvidia.com/zh-cn/cuda-gpus#compute 页面查找如下：

![image](https://user-images.githubusercontent.com/104058290/196312385-c4bbc182-7eb4-4f1f-a9e5-a370f8227ca6.png)

虽然没有找到A6000，不过我通过其他方法查到
A6000对应sm_86

第二步，匹配对应算力的Cuda版本
https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html#application-compatibility-on-ampere
![image](https://user-images.githubusercontent.com/104058290/196311128-9a098f5e-32c0-4a18-bbed-9194faf137fa.png)

可以看到cuda11.0+才支持sm_80（包括sm_86）
保险起见，搜索相关论坛讨论，最低采用cuda11.3版本，如前所述，最高11.7

第三步，安装cuda
好吧，其实是叫你不要安装。
等解决所有涉及版本，再装不迟

# 2. 选择cuDNN
假设通过1.0确定cuda11.3，到官网 https://developer.nvidia.com/rdp/cudnn-archive#a-collapse805-111 下载与CUDA匹配的cuDNN

![image](https://user-images.githubusercontent.com/104058290/196313825-ac47e8f2-8473-4a22-b282-5791e3e7b115.png)

下载成功等之后一并安装

# 3. 选择Pytorch(GPU)和python
根据前面确定的cuda，选择pytorch版本。
在 https://download.pytorch.org/whl/torch_stable.html 页面寻找torch（GPU）与python对应资源

![image](https://user-images.githubusercontent.com/104058290/196314766-f53f0c6c-63e5-432a-b523-85ec4491d1f7.png)

cu113代表为匹配cuda11.3的GPU版本；torch-1.10.* 代表1.10.* 版本的pytorch；cp-38代表匹配python3.8版本；Linux_x86_64为匹配系统架构





