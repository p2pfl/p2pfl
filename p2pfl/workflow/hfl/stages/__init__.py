#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""HFL workflow stages."""

from p2pfl.workflow.hfl.stages.edge_aggregate_workers import HFLEdgeAggregateWorkersStage
from p2pfl.workflow.hfl.stages.edge_distribute import HFLEdgeDistributeStage
from p2pfl.workflow.hfl.stages.edge_local_train import HFLEdgeLocalTrainStage
from p2pfl.workflow.hfl.stages.edge_sync_root import HFLEdgeSyncRootStage
from p2pfl.workflow.hfl.stages.root_aggregate import HFLRootAggregateStage
from p2pfl.workflow.hfl.stages.root_distribute import HFLRootDistributeStage
from p2pfl.workflow.hfl.stages.round_finished import HFLRoundFinishedStage
from p2pfl.workflow.hfl.stages.setup import HFLSetupStage
from p2pfl.workflow.hfl.stages.worker_train import HFLWorkerTrainStage
from p2pfl.workflow.hfl.stages.worker_wait_global import HFLWorkerWaitGlobalStage

__all__ = [
    "HFLSetupStage",
    "HFLWorkerTrainStage",
    "HFLWorkerWaitGlobalStage",
    "HFLEdgeLocalTrainStage",
    "HFLEdgeAggregateWorkersStage",
    "HFLEdgeSyncRootStage",
    "HFLEdgeDistributeStage",
    "HFLRootAggregateStage",
    "HFLRootDistributeStage",
    "HFLRoundFinishedStage",
]
