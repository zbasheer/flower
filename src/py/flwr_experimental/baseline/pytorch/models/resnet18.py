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
"""Implementation of ResNet18 using Group Normalization[1] for CIFAR-10/100 as
seen in:

- K. Hsieh, A. Phanishayee, O. Mutlu, and P. B. Gibbons,
    'The non-iid data quagmire of decentralized machine learning,' arXiv, no. Ml, 2019.
- S. J. Reddi et al., “Adaptive federated optimization,” arXiv, no. 2, pp. 1–40, 2020.

[1] Y. Wu and K. He, “Group Normalization,” Int. J. Comput. Vis., vol. 128, no. 3,
    pp. 742–755, Mar. 2020, doi: 10.1007/s11263-019-01198-w.
"""
from collections import OrderedDict
from typing import Any, Union

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.normalization import GroupNorm
from torch.utils.data import Dataset
from torchvision.models import resnet18

import flwr as fl


class Net(nn.Module):  # type: ignore
    """Simple Resnet18 using GroupNorm instead of BatchNorm."""

    def __init__(self, num_classes:int) -> None:
        super(Net, self).__init__()
        self.model = resnet18(norm_layer=lambda x: GroupNorm(2, x))
        self.model.fc =nn.Linear(512, num_classes) 

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        return self.model(x)

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict, strict=True)


def load_model(
    num_classes: int = 10,
) -> Union[nn.Module, nn.ModuleList, nn.ModuleDict, Net]:
    """Returns a ResNet18 model with group normalization layers."""
    resnet18gn = Net(num_classes=num_classes)
    return resnet18gn
