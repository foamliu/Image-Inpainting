# 图像补绘

基于反卷积网络实现图像补绘

## 原理



## 数据集

![image](https://github.com/foamliu/Image-Impainting/raw/master/images/imagenet.png)

按照 [说明](https://github.com/foamliu/ImageNet-Downloader) 下载 ImageNet 数据集。

## 如何使用


### 训练
```bash
$ python train.py
```

如果想可视化训练效果，请运行:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### 演示

下载预训练的 [补绘模型](https://github.com/foamliu/MDSR/releases/download/v1.0/model.16-21.4264.hdf5)，放入 "models" 目录。然后执行:


```bash
$ python demo.py
```