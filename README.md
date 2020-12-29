# NGCF-Pytorch
Pytorch implementation of NGCF

This is a pytorch version of [NGCF](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)(Neural Graph Collaborative Filtering).

We have used the same data processing code(data_process.py) of NGCF, thanks.

# Environment Requirement
The code has been tested under python 3.7.6. The required packages are as follows:
* pytorch == 1.0.0
* numpy == 1.18.1
* scipy == 1.4.1

# Run the demo
```
python NGCF_recommendation.py
```
This code is using the gpu with the help of pytorch, but the training process is really a long time. 

And I can't find out that why pytorch version slower so much compared with tensorflow version. I would apperacite a lot if you can issue it.


# The result is:
```
400 epochs training: pre@10 0.06327 recall@10 0.10312, pre@20 0.04485 recall@20 0.14426
```

If this repo helps you, please star.(This will help me!)

uid 18736: pre@10 0.04582 recall@10 0.06161, pre@20 0.03526 recall@20 0.09270 ndcg@20 0.04278