### 运行环境

```
pip install -r requirements.txt
```

### 训练

在没有显示器的服务器上运行命令

```
xvfb-run -s '-screen 0 1024x768x24' python3 train.py 
```

在有显示器的电脑上运行

```
python3 train.py 
```

训练完了以后模型会保存在`train/` 文件夹下面，直接运行的训练时常大概是3天左右（72h）（硬件为1x 2080Ti + 4xCPU）。

### 测试

与训练一样，没有显示器的话需要在前面加上一些命令：

```
xvfb-run -s '-screen 0 1024x768x24' python3 test.py
python3 test.py
```

正常训练结束以后，测试的平均reward大概在$30-40$ 之间。

### 相关链接

- Docs: [http://www.minerl.io/docs/](http://www.minerl.io/docs/)
- Github: [https://github.com/minerllabs/minerl](https://github.com/minerllabs/minerl)
- AIcrowd: [https://www.aicrowd.com/challenges/neurips-2019-minerl-competition](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition)
- Competition Proposal: [https://arxiv.org/abs/1904.10079](https://arxiv.org/abs/1904.10079)
- 数据集: https://router.sneakywines.me/minerl_v1/data_texture_0_low_res.tar.gz
- 额外的数据集：https://router.sneakywines.me/minerl-v123321123321/data_texture_0_low_res.tar.gz