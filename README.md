# tow

### info
Implementing ByteT5 model and trainer using PyTorch. Exploring various position encoding methods (ALiBi, RoPE) and UTF-8-Unicode encoding.


### run
```
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 nohup torchrun --standalone --nproc_per_node=6 train.py test_models/byt5-base test &

```