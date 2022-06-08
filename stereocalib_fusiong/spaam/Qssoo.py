# -*- python coding: utf-8 -*-
# @Time: 5月 17, 2022
# ---
import torch
import torch.nn.quantized as nnq

x = torch.Tensor([[[[-1, -2, -3], [1, 2, 3]]]])
print(x)
xq = torch.quantize_per_tensor(x, scale=0.0472, zero_point=64, dtype=torch.quint8)
print(xq)
print("xq", xq.int_repr())
print("------------------------------------")

c = nnq.Conv2d(1, 1, 1)
weight = torch.Tensor([[[[-0.7898]]]])
qweight = torch.quantize_per_channel(weight, scales=torch.Tensor([0.0062]).to(torch.double),
                                     zero_points=torch.Tensor([0]).to(torch.int64), axis=0, dtype=torch.qint8)
print(qweight)
y1 = xq.int_repr() * qweight.int_repr()
print(y1)
y2 = qweight.int_repr().type(torch.int16) * 64
print(y2)

print("-****---", torch.round(((y1 - y2) * 0.0062 * 0.0472/0.03714831545948982 + 64)))

c.set_weight_bias(qweight, None)
c.scale = 0.03714831545948982
c.zero_point = 64
xx = c(xq)
print(xx)
print(xx.int_repr())
