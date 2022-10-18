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
等解决所有涉及版本，再装不迟。
不过可以先下载，也可以最后返回下载。在cuda官网 https://developer.nvidia.cn/cuda-toolkit-archive 选择合适的版本。



我们选择11.3.0，点击转到下载页面，选择对应系统版本：

![image](https://user-images.githubusercontent.com/104058290/196317456-32d0e4fc-7738-4885-ba25-1f7d3001b79d.png)

我们的Ubuntu为22.04，选择20.04勉强能用。出现下载/运行命令：

![image](https://user-images.githubusercontent.com/104058290/196317874-202ed43c-b09f-4075-9e37-533a2904d03a.png)

```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```
存起来最后安装再用

# 2. 选择cuDNN
假设通过1.0确定cuda11.3，到官网 https://developer.nvidia.com/rdp/cudnn-archive#a-collapse805-111 下载与CUDA匹配的cuDNN

![image](https://user-images.githubusercontent.com/104058290/196313825-ac47e8f2-8473-4a22-b282-5791e3e7b115.png)

下载成功等之后一并安装

# 3. 选择Pytorch(GPU)和python
根据前面确定的cuda，选择pytorch版本。
在 https://download.pytorch.org/whl/torch_stable.html 页面寻找torch（GPU）与python对应资源

![image](https://user-images.githubusercontent.com/104058290/196314766-f53f0c6c-63e5-432a-b523-85ec4491d1f7.png)

cu113代表为匹配cuda11.3的GPU版本；torch-1.10.* 代表1.10.* 版本的pytorch；cp-38代表匹配python3.8版本；Linux_x86_64为匹配系统架构
！！！切记保证所选版本之间匹配，存在，如上图所示pytorch-1.11.0已经不支持python-3.6

我们选择需要的 cuda11.3 torch-1.10.0 python3.8 linux_x86_64 对应版本

# 4. 安装
# 4.1. 安装Cuda（由ROOT用户操作）
cuda安装在系统目录中，方便所有用户使用，也可以避免安装中的麻烦，因此由root用户安装。
第一步，随便在哪个目录下（记得安装完删掉安装程序就行），使用1.中得到的cuda命令：
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```
安装成功后（以cuda11.5为例）：

![dc92757ad4a148d3ae11575f6a4c6c1c](https://user-images.githubusercontent.com/104058290/196319828-532bfec4-d92b-4d34-b281-9cd9c98bf0e3.png)

！记住图中"Please make sure that" 中的两个路径：/usr/local/cuda-11.* 下面的bin和lib64（有的为lib，没有64）

第二步，配置环境变量（由非ROOT用户配置）
每个用户需要使用不同的cuda，用户根据需要，配置自己的环境变量调用。
打开配置文件（有的使用的zsh，非bash，不考虑此情况）：

```
vim ~/.bashrc
```

在配置文件末尾加上（注意这里的/usr/local/cuda-11.* 路径和上面安装截图里面的路径一致）：

```
export PATH=//usr/local/cuda-11.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64$LD_LIBRARY_PATH
```

保存退出，激活配置文件：

```
source ~/.bashrc
```

使用```nvcc -V```检查CUDA是否安装成功，出现提示代表安装成功。

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:15:46_PDT_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0
```
# 4.2. 安装cuDNN（由ROOT用户操作）
转到2.0中cudnn-11.3-linux-x64-v8.2.0.53下载的目录,解压cuDNN文件
```
  tar -xvf cudnn-11.3-linux-x64-v8.2.0.53.tgz
```
并进入解压出来的文件```ls```,结果如下：
```include  lib64  NVIDIA_SLA_cuDNN_Support.txt```
拷贝文件到/usr/local/cuda-11.3(上一步cuda安装路径）中：
```
  sudo cp lib64/* /usr/local/cuda-11.3/lib64/
	sudo cp include/* /usr/local/cuda-11.3/include/
	sudo chmod a+r /usr/local/cuda-11.3/lib64/*
	sudo chmod a+r /usr/local/cuda-11.3/include/*
```
查看cuDNN版本，```cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2```

```
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 2
#define CUDNN_PATCHLEVEL 0
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#endif /* CUDNN_VERSION_H */
```

安装成功





