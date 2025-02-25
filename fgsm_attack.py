import torch, os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from resnet import ResNet50
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Cifar_10_Dataset(Dataset):
    def __init__(self, data_path, label_path):
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3)
        var = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3)
        self.x = np.load(data_path) / 255
        self.x = (self.x - mean) / var
        self.x = self.x.transpose(0, 3, 1, 2)
        self.label = np.load(label_path)
        self.label = np.reshape(self.label, (self.x.shape[0], ))

        # https://stackoverflow.com/questions/44717100/pytorch-convert-floattensor-into-doubletensor
        self.x, self.label = torch.from_numpy(
            self.x).float(), torch.from_numpy(self.label)

    def __getitem__(self, index):
        return self.x[index], self.label[index]

    def __len__(self):
        return self.x.size(0)


def fgsm_attack(image, epsilon, data_grad):
    # sign 运算，正数为 1，负数为 0
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


print('load model')
model = ResNet50()
pth_file = 'resnet50_ckpt.pth'

d = torch.load(pth_file)['net']
d = OrderedDict([(k[7:], v) for (k, v) in d.items()])
model.load_state_dict(d)
model.to(device)
model.eval()

import os
import numpy as np
import pickle

# CIFAR-10 原始数据存放目录
cifar10_dir = "data/cifar-10-batches-py"  # 确保该目录存在
save_dir = "data/cifar10"  # 目标目录
os.makedirs(save_dir, exist_ok=True)

# 加载 CIFAR-10 数据
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # 归一化
    labels = np.array(batch[b'labels'])
    return data, labels

# 处理 **训练集**（多个 batch）
data_list = []
label_list = []
for i in range(1, 6):
    batch_data, batch_labels = load_cifar_batch(os.path.join(cifar10_dir, f"data_batch_{i}"))
    data_list.append(batch_data)
    label_list.append(batch_labels)

# 处理 **测试集**
test_data, test_labels = load_cifar_batch(os.path.join(cifar10_dir, "test_batch"))

# **合并训练集和测试集**
all_data = np.concatenate(data_list + [test_data])  # (60000, 3, 32, 32)
all_labels = np.concatenate(label_list + [test_labels])  # (60000,)
all_data = all_data.transpose(0, 2, 3, 1)  # (60000, 32, 32, 3)

# **保存 `.npy` 文件**
np.save(os.path.join(save_dir, "cifar10_data.npy"), all_data)
np.save(os.path.join(save_dir, "cifar10_label.npy"), all_labels)

print(f"✅ CIFAR-10 全部数据（60000 张图片）已转换完成，存放在 {save_dir}")


data_path = "data/cifar10/cifar10_data.npy"
label_path = "data/cifar10/cifar10_label.npy"

print('load data')
data_set = Cifar_10_Dataset(data_path=data_path, label_path=label_path)
data_loader = DataLoader(dataset=data_set, batch_size=128, shuffle=False)

for epsilon in [0.05, 0.1, 0.15, 0.3]:
    save_data = None
    print("{} attack...".format(epsilon))
    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        target = target.long()

        loss = F.nll_loss(output, target)
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        if save_data is None:
            save_data = perturbed_data.detach().cpu().numpy()
        else:
            save_data = np.concatenate(
                (save_data, perturbed_data.detach().cpu().numpy()), axis=0)
    np.save('fgsm_adv_examples/{}_cifar10.npy'.format(epsilon),
            save_data)
    print('{}_cifar10 has been saved'.format(epsilon))
