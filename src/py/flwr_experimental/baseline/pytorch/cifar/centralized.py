# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Centralized training for CIFAR-10/100."""
from typing import Dict, Tuple, Type

import torch
from torch import LongTensor, Tensor, device, nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm

from ..dataloader.cifar import CIFAR_PartitionedDataset, get_normalization_transform
from ..models.resnet18 import load_model
from . import DATA_ROOT


def train(
    *,
    net: nn.Module,
    dataloader: "DataLoader[Tuple[Tensor, LongTensor]]",
    num_epochs: int,
    device: torch.device,  # pylint: disable=no-member
    optim_type: Type[torch.optim.Optimizer] = torch.optim.SGD,
    optim_defaults: Dict[str, float] = {"lr": 0.01},
) -> None:
    """Train the network."""
    net.train()
    # Define loss and optimizer
    criterion = nn.modules.loss.CrossEntropyLoss()
    net = net.to(device)
    optimizer = optim_type(net.parameters(), **optim_defaults)

    # Train the network
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Training epoch {epoch} out of {num_epochs}")
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    *,
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    net.eval()
    net = net.to(device)
    criterion = nn.modules.loss.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    print("Evaluating...")
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(
                input=outputs.data, dim=1
            )  # pylint: disable-msg=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Loss: {loss}  Accuracy: {accuracy}")
    return loss, accuracy


if __name__ == "__main__":

    net = load_model(num_classes=10)
    # net = resnet18(norm_layer=lambda x: nn.GroupNorm(2, x))
    trainset = CIFAR_PartitionedDataset(
        num_classes=10,
        root_dir="~/.flower/data/cifar10/partitions/lda/0.10/train",
        partition_id=0,
        transform=get_normalization_transform(),
    )
    """trainset = CIFAR10(
        root=DATA_ROOT,
        download=True,
        train=True,
        transform=get_normalization_transform(),
    )"""
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2
    )

    testset = CIFAR10(
        root=DATA_ROOT,
        download=True,
        train=False,
        transform=get_normalization_transform(),
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=True, num_workers=2
    )

    writer = SummaryWriter("centralized")
    device = torch.device("cuda:0")
    loss, accuracy = test(net=net, dataloader=testloader, device=device)
    print(f"Initial Loss: {loss:.2f} | Initial accuracy: {accuracy:.2f}")
    for epoch, _ in enumerate(tqdm(range(30))):
        train(net=net, dataloader=trainloader, num_epochs=1, device=device)
        loss, accuracy = test(net=net, dataloader=testloader, device=device)
        writer.add_scalar("Loss/test", loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        print(f"Accuracy {accuracy} after {epoch} epochs. Loss: {loss:.2f}")
