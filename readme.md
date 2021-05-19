> 本项目参考 [Github](https://github.com/TingsongYu/ghostnet_cifar10)
## 文件说明
> 01_parse_cifar10_to_png.py 解压cifar10
> 02_main.py 运行模型
> 03 计算模型复杂度
> 04 转换pkl文件为pt文件
> config.py 配置运行文件
## 使用配置
### 数据集

> 虫子数据集按如下图所示

![1](/pic/1.png)

### 虫子类

在config中更改class_names

定制train_bs与valid_bs

![1](/pic/2.png)

使用合适的数据预处理和增强

![1](/pic/3.png)

## 训练命令

>训练resnet56： python bin/02_main.py -gpu 0 -arc resnet56 
>训练ghost-resnet56：python bin/02_main.py -gpu 0 -arc resnet56 -replace_conv
>训练 vgg16：python bin/02_main.py -gpu 0 -arc vgg16
>训练ghost-vgg16：python bin/02_main.py -gpu 0 -arc vgg16 -replace_conv 



