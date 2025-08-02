import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as T
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchinfo import summary

torch.manual_seed(0)
np.random.seed(0)

# Model
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(3*32*32, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Dataset
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.ToPureTensor(),
])

dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

n_val = int(len(dataset_test)*0.2)
n_test = len(dataset_test) - n_val

dataset_test, dataset_val = torch.utils.data.random_split(dataset_test, [n_test, n_val], generator=torch.Generator().manual_seed(42))
print(f"Train: {len(dataset_train)}")
print(f"Val  : {len(dataset_val)}")
print(f"Test : {len(dataset_test)}")

id2cls = {i: cls for i, cls in enumerate(dataset_train.classes)}
print(id2cls)

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

# Results directory
results_dir = Path("results")

# Model
model = MLP(num_classes=10)
model = model.to(device)
model.load_state_dict(torch.load(results_dir/"best.pth", weights_only=True))
model.eval()

# Evaluation - Accuracy
criterion = nn.CrossEntropyLoss()

acc = Accuracy(task="multiclass", num_classes=10)
acc = acc.to(device)

acc_per_cls = Accuracy(task="multiclass", num_classes=10, average=None)
acc_per_cls = acc_per_cls.to(device)

confmat = ConfusionMatrix(task="multiclass", num_classes=10)
confmat = confmat.to(device)

test_loss = 0
test_preds = []
test_labels = []
for batch in tqdm(dataloader_test):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    with torch.inference_mode():
        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        acc.update(preds, labels)
        acc_per_cls.update(preds, labels)
        confmat.update(preds, labels)
        test_loss += loss.item()

        test_preds.append(preds.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

test_loss = test_loss / len(dataloader_test)
test_acc = acc.compute()
acc.reset()
print("(Test)  loss {:.4f}, accuracy {:.4f}".format(test_loss, test_acc))

test_preds = np.concatenate(test_preds)
test_labels = np.concatenate(test_labels)

# Evaluation - Accuracy per class, Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
acc_per_cls.plot(ax=axes[0])
confmat.plot(ax=axes[1])
fig.savefig(results_dir/"eval.png")
plt.close(fig)

# Visual evaluation
n_to_show = 10
indices = np.random.choice(range(len(dataset_test)), n_to_show)

fig = plt.figure(figsize=(20, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = dataset_test[idx][0].permute(1, 2, 0)
    true_label = id2cls[test_labels[idx]]
    pred_label = id2cls[test_preds[idx]]

    if true_label == pred_label:
        color = "green"
    else:
        color = "red"

    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        "true = " + true_label,
        fontsize=10,
        color=color,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.7,
        "pred = " + pred_label,
        fontsize=10,
        color=color,
        ha="center",
        transform=ax.transAxes,
    )
    ax.imshow(img)
fig.savefig(results_dir/"visual_eval.png")
plt.close(fig)