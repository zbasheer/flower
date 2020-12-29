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
"""Adaptive Federated Optimization using Yogi (FedYogi) [Reddi et al., 2020]
strategy.

Paper: https://arxiv.org/abs/2003.00295
"""


from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg
from .fedopt import FedOpt


class FedYogi(FedOpt):
    """Configurable FedYogi strategy implementation."""

    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        accept_failures: bool = True,
        current_weights: Weights,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            current_weights=current_weights,
            eta=eta,
            eta_l=eta_l,
            tau=tau,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta_t = np.zeros_like(self.current_weights)
        self.v_t = np.zeros_like(self.current_weights)

    def __repr__(self) -> str:
        rep = f"FedYogi(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        fedavg_aggregate = super().aggregate_fit(
            rnd=rnd, results=results, failures=failures
        )
        if fedavg_aggregate is None:
            return None
        aggregated_updates = fedavg_aggregate[0] - self.current_weights[0]

        # Yogi
        self.delta_t = (
            self.beta_1 * self.delta_t + (1.0 - self.beta_1) * aggregated_updates
        )
        delta_t_sq = np.multiply(self.delta_t, self.delta_t)
        self.v_t = self.v_t - (1.0 - self.beta_2) * delta_t_sq * np.sign(
            self.v_t - delta_t_sq
        )

        weights = [
            self.current_weights[0]
            + self.eta * self.delta_t / (np.sqrt(self.v_t) + self.tau)
        ]

        return weights
