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

输入 | 输出 | 目标 | 
|---|---|---|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/0_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/0_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/0_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/1_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/1_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/1_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/2_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/2_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/2_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/3_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/3_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/3_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/4_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/4_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/4_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/5_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/5_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/5_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/6_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/6_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/6_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/7_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/7_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/7_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/8_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/8_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/8_gt.png)|
|![image](https://github.com/foamliu/Image-Impainting/raw/master/images/9_input.png) | ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/9_output.png)| ![image](https://github.com/foamliu/Image-Impainting/raw/master/images/9_gt.png)|
