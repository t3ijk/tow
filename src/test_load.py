import torch

model = torch.load('model.txt')
print(type(model))
for item in model:
    print(item)
    # print(item[0])
