# NIPS 2019：MineRL Competition

A pytorch solution for 4th place [NIPS 2019 MineRL Competition](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition/leaderboards?challenge_round_id=126&post_challenge=on)

## How to run it

#### runtime environment

```
pip install -r requirements.txt
```

Once you install this requirements, you can run our code. **This solution did not use human datasets**.

#### training

Run minerl environments **without a head(servers without monitors)** use a software renderer such as xvfb：

```
xvfb-run -s '-screen 0 1024x768x24' python3 train.py 
```

Run mineral environments **with a head(servers without monitors attached)**:

```
python3 train.py 
```

The model will be saved in this this folder `train/` when training complete. The training process takes **72h**(1x2080Ti + 4xCPU) .

####testing

Run it just like training:

```
xvfb-run -s '-screen 0 1024x768x24' python3 test.py
python3 test.py
```

The average reward of test is **between 30 and 40** after normal training. 

##Related Links

- Docs: [http://www.minerl.io/docs/](http://www.minerl.io/docs/)
- Github: [https://github.com/minerllabs/minerl](https://github.com/minerllabs/minerl)
- AIcrowd: [https://www.aicrowd.com/challenges/neurips-2019-minerl-competition](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition)
- Competition Proposal: [https://arxiv.org/abs/1904.10079](https://arxiv.org/abs/1904.10079)
- Human datasets: https://router.sneakywines.me/minerl_v1/data_texture_0_low_res.tar.gz
- Extra human datasets：https://router.sneakywines.me/minerl-v123321123321/data_texture_0_low_res.tar.gz