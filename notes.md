### 1 attention_outs`s variance is increasing?

```log
v_output torch.Size([1, 5, 1472]) (tensor(20.4425, grad_fn=<VarMeanBackward0>), tensor(0.0103, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(38.6018, grad_fn=<VarMeanBackward0>), tensor(-0.2179, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(40.0511, grad_fn=<VarMeanBackward0>), tensor(0.2390, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(45.4637, grad_fn=<VarMeanBackward0>), tensor(0.0804, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(28.7543, grad_fn=<VarMeanBackward0>), tensor(0.0122, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(110.1162, grad_fn=<VarMeanBackward0>), tensor(-0.0043, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(243.2697, grad_fn=<VarMeanBackward0>), tensor(0.0088, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(482.8038, grad_fn=<VarMeanBackward0>), tensor(0.3676, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(1032.9785, grad_fn=<VarMeanBackward0>), tensor(-0.7416, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(1521.9875, grad_fn=<VarMeanBackward0>), tensor(-0.3669, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(3920.5374, grad_fn=<VarMeanBackward0>), tensor(-2.2470, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 5, 1472]) (tensor(6732.2520, grad_fn=<VarMeanBackward0>), tensor(-0.1261, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(37.5800, grad_fn=<VarMeanBackward0>), tensor(-0.1193, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(18.0383, grad_fn=<VarMeanBackward0>), tensor(0.1398, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(20.7326, grad_fn=<VarMeanBackward0>), tensor(0.0785, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(166.8336, grad_fn=<VarMeanBackward0>), tensor(-0.2200, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(27.8930, grad_fn=<VarMeanBackward0>), tensor(-0.0036, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(459.7788, grad_fn=<VarMeanBackward0>), tensor(-0.4172, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(262.5247, grad_fn=<VarMeanBackward0>), tensor(-0.0647, grad_fn=<VarMeanBackward0>))
v_output torch.Size([1, 3, 1472]) (tensor(2829.0522, grad_fn=<VarMeanBackward0>), tensor(-1.2726, grad_fn=<VarMeanBackward0>))
tensor(11.7353, grad_fn=<NllLossBackward0>) deltaT 0.4409174919128418
```

### 2 some logits`s variance is inf?  because masked bias is infinity.

```log
$ python test_byt5.py 
logits torch.Size([1, 6, 5, 5]) (tensor(44.5965, grad_fn=<VarMeanBackward0>), tensor(0.4450, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(310.0781, grad_fn=<VarMeanBackward0>), tensor(16.5432, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(1003.2698, grad_fn=<VarMeanBackward0>), tensor(25.9056, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(153.7862, grad_fn=<VarMeanBackward0>), tensor(8.9434, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(75.1691, grad_fn=<VarMeanBackward0>), tensor(5.9649, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(32.9662, grad_fn=<VarMeanBackward0>), tensor(2.7413, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(40.1116, grad_fn=<VarMeanBackward0>), tensor(1.7433, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(50.1414, grad_fn=<VarMeanBackward0>), tensor(2.2026, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(38.9291, grad_fn=<VarMeanBackward0>), tensor(0.7331, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(44.9935, grad_fn=<VarMeanBackward0>), tensor(-0.8383, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(48.9146, grad_fn=<VarMeanBackward0>), tensor(-2.1540, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 5, 5]) (tensor(42.8501, grad_fn=<VarMeanBackward0>), tensor(-0.7823, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 3]) (tensor(inf, grad_fn=<VarMeanBackward0>), tensor(-1.1343e+38, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 5]) (tensor(6.1122, grad_fn=<VarMeanBackward0>), tensor(-2.7927, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 3]) (tensor(inf, grad_fn=<VarMeanBackward0>), tensor(-1.1343e+38, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 5]) (tensor(3.6693, grad_fn=<VarMeanBackward0>), tensor(-0.0689, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 3]) (tensor(inf, grad_fn=<VarMeanBackward0>), tensor(-1.1343e+38, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 5]) (tensor(3.2478, grad_fn=<VarMeanBackward0>), tensor(-0.1657, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 3]) (tensor(inf, grad_fn=<VarMeanBackward0>), tensor(-1.1343e+38, grad_fn=<VarMeanBackward0>))
logits torch.Size([1, 6, 3, 5]) (tensor(1.6484, grad_fn=<VarMeanBackward0>), tensor(-0.7373, grad_fn=<VarMeanBackward0>))
tensor(11.7353, grad_fn=<NllLossBackward0>) deltaT 0.39883923530578613
```