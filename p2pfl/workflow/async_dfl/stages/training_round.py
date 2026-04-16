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
"""
Training round stage for AsyncDFL.

Implements the asynchronous decentralized federated learning loop:
debiasing -> train_on_batch -> broadcast_loss -> [network_update] -> round_finish.
"""

from __future__ import annotations

import math

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.workflow.async_dfl.context import AsyncDFLContext
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.evaluate import evaluate_and_broadcast
from p2pfl.workflow.shared.gossiping import ModelGate, should_accept_model


class TrainingRoundStage(Stage[AsyncDFLContext]):
    """Debiasing, training, network updating, and round finishing stage."""

    async def run(self) -> str | None:
        """Execute one training round: debias, train, gossip, aggregate."""
        ctx = self.ctx
        address = ctx.address
        learner = ctx.learner
        experiment = ctx.experiment

        # Phase 1: Debias model with push-sum weight
        await self._debias_model(ctx)

        # Phase 2: Train on a single batch
        logger.info(address, "🏋️‍♀️ Updating local model...")
        trained_model = await learner.train_on_batch()
        peer = ctx.peers.get(address)
        if peer is None:
            logger.warning(address, f"Local peer state not found for {address}")
        else:
            peer.model = trained_model

        # Phase 3: Broadcast training loss
        await self._broadcast_loss(ctx)

        # Phase 4: Network update (every tau rounds)
        if experiment.round > 0 and experiment.round % experiment.tau == 0:
            await self._network_update(ctx)

        # Phase 5: Round finish
        experiment.round += 1
        for p in ctx.peers.values():
            p.reset_round()
        logger.info(address, f"Round {experiment.round} finished.")

        # Check termination
        if ctx.experiment.is_complete():
            return "finish"
        return "training_round"

    ###
    #    Phase 1: Debiasing
    ###

    async def _debias_model(self, ctx: AsyncDFLContext) -> None:
        """Apply push-sum debiasing to the local model."""
        logger.debug(ctx.address, "Debiasing model.")
        model = await ctx.learner.aget_model()
        peer = ctx.peers.get(ctx.address)
        if peer is not None and hasattr(model, "set_push_sum_weight"):
            model.set_push_sum_weight(peer.push_sum_weight)
            await ctx.learner.aset_model(model)

    ###
    #    Phase 3: Loss broadcasting
    ###

    async def _broadcast_loss(self, ctx: AsyncDFLContext) -> None:
        """Broadcast the current training loss to all peers."""
        model = await ctx.learner.aget_model()
        training_loss = getattr(model, "last_training_loss", 0.0)
        peer = ctx.peers.get(ctx.address)
        if peer is None:
            logger.warning(ctx.address, f"Local peer state not found for {ctx.address}")
        else:
            peer.add_loss(ctx.experiment.round, training_loss)

        logger.info(ctx.address, "📡 Broadcasting loss values.")
        try:
            await ctx.cp.broadcast_gossip(
                ctx.cp.build_msg(
                    "loss_information_updating",
                    [str(training_loss)],
                    round=ctx.experiment.round,
                )
            )
        except Exception as e:
            logger.warning(ctx.address, f"Failed to broadcast loss: {e}")

    ###
    #    Phase 4: Network update
    ###

    async def _network_update(self, ctx: AsyncDFLContext) -> None:
        """Compute priorities, select neighbors, gossip, and aggregate."""
        address = ctx.address

        # Compute priorities and select neighbors
        neighbor_priorities = self._compute_priorities(ctx)
        logger.info(address, f"Neighbor priorities: {neighbor_priorities}")
        ctx.candidates = self._select_neighbors(neighbor_priorities, top_k=ctx.experiment.top_k_neighbors)
        logger.info(address, f"Selected neighbors: {ctx.candidates}")

        # Gossip model to selected neighbors
        await self._gossip_model(ctx)

        # Aggregate received models
        await self._aggregate(ctx)

    def _compute_priorities(self, ctx: AsyncDFLContext) -> list[tuple[str, float]]:
        """Compute priority for each neighbor based on loss divergence and staleness."""
        peers = ctx.peers
        tau = ctx.experiment.tau
        neighbor_priorities: list[tuple[str, float]] = []

        local_peer = peers.get(ctx.address)

        for neighbor in list(ctx.cp.get_neighbors(only_direct=True)):
            neighbor_peer = peers.get(neighbor)
            if neighbor_peer is None:
                continue
            t_hat = min(ctx.experiment.round, neighbor_peer.round_number)
            local_loss = _windowed_avg_loss(local_peer.losses, t_hat, tau) if local_peer else 0.0
            neighbor_loss = _windowed_avg_loss(neighbor_peer.losses, t_hat, tau)
            priority = compute_priority(
                ti=ctx.experiment.round,
                tp_ij=neighbor_peer.push_time,
                tj=neighbor_peer.round_number,
                tl_ji=neighbor_peer.p2p_updating_idx,
                f_ti=local_loss,
                f_tj=neighbor_loss,
                dmax=ctx.experiment.dmax,
            )
            neighbor_priorities.append((neighbor, priority))

        return neighbor_priorities

    @staticmethod
    def _select_neighbors(neighbor_priorities: list[tuple[str, float]], top_k: int = 3) -> list[str]:
        """Select the top-k neighbors by priority."""
        ranked = sorted(neighbor_priorities, key=lambda x: x[1], reverse=True)
        return [n for n, _ in ranked[:top_k]]

    async def _gossip_model(self, ctx: AsyncDFLContext) -> None:
        """Send model and push-sum weight to selected neighbors."""
        address = ctx.address
        learner = ctx.learner
        experiment = ctx.experiment
        model = await learner.aget_model()

        gate = ModelGate(ctx.cp, address, pre_send_command="pre_send_model_training")

        for neighbor in ctx.candidates:
            round_num = experiment.round
            payload = ctx.cp.build_weights(
                "model_information_updating",
                round_num,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            sent = await gate.send_if_accepted(
                neighbor=neighbor,
                weight_command="model_information_updating",
                contributors=model.get_contributors(),
                round_num=round_num,
                payload=payload,
            )
            if sent:
                await self._send_push_sum_weight(ctx, neighbor)
                neighbor_peer = ctx.peers.get(neighbor)
                if neighbor_peer is not None:
                    neighbor_peer.push_time = round_num
                else:
                    logger.warning(address, f"Peer state not found for {neighbor}")

    async def _send_push_sum_weight(self, ctx: AsyncDFLContext, neighbor: str) -> None:
        """Send the local push-sum weight to a neighbor."""
        self_peer = ctx.peers.get(ctx.address)
        push_sum_weight = self_peer.push_sum_weight if self_peer else 1.0

        try:
            await ctx.cp.send(
                nei=neighbor,
                msg=ctx.cp.build_msg(
                    "push_sum_weight_information_updating",
                    [push_sum_weight],
                    round=ctx.experiment.round,
                ),
            )
        except Exception as e:
            logger.warning(ctx.address, f"Failed to send push-sum weight to {neighbor}: {e}")

    async def _aggregate(self, ctx: AsyncDFLContext) -> None:
        """Prepare models for aggregation and update protocol state."""
        address = ctx.address

        # Attach mixing/push-sum info and notify in-neighbors
        models = []
        for neighbor, peer in ctx.peers.items():
            if peer.model is None:
                continue
            peer.model.add_info("mixing_weight", peer.mixing_weight)
            peer.model.add_info("push_sum_weight", peer.push_sum_weight)
            models.append(peer.model)

            if neighbor != address:
                peer.p2p_updating_idx = ctx.experiment.round
                try:
                    await ctx.cp.send(
                        nei=neighbor,
                        msg=ctx.cp.build_msg("index_information_updating", round=ctx.experiment.round),
                    )
                except Exception as e:
                    logger.warning(address, f"Failed to send iteration index to {neighbor}: {e}")

        # Aggregate (Eq. 5 & 6 handled by PushSum aggregator)
        if models:
            agg_model = ctx.aggregator.aggregate(models)
            await ctx.learner.aset_model(agg_model)

            # Update local push-sum weight from aggregator result
            self_peer = ctx.peers.get(address)
            if self_peer is not None:
                self_peer.push_sum_weight = agg_model.get_info().get("push_sum_weight", self_peer.push_sum_weight)

        await evaluate_and_broadcast(ctx)
        logger.info(address, "Aggregation finished.")

    ###
    #    Message handlers
    ###

    @on_message("loss_information_updating")
    async def handle_loss_information(self, source: str, round: int, *args) -> None:
        """Handle a loss_information_updating message."""
        if not args:
            raise ValueError("Loss value is required")
        loss = float(args[0])
        peer = self.ctx.peers.get(source)
        if peer is None:
            logger.warning(self.ctx.address, f"Peer state not found for {source}")
            return
        try:
            peer.add_loss(round, loss)
            logger.debug(self.ctx.address, f"{source} loss updated to {loss} for round {round}")
        except Exception as e:
            logger.error(self.ctx.address, f"Error saving loss from {source} for round {round}: {e}")

    @on_message("index_information_updating")
    async def handle_index_information(self, source: str, round: int, *args) -> None:
        """Handle an index_information_updating message."""
        peer = self.ctx.peers.get(source)
        if peer is None:
            logger.warning(self.ctx.address, f"Peer state not found for {source}")
            return
        try:
            peer.round_number = round
            logger.debug(self.ctx.address, f"{source} round updated to {round}")
        except Exception as e:
            logger.error(self.ctx.address, f"Error saving iteration index from {source}: {e}")

    @on_message("model_information_updating", weights=True)
    async def handle_model_information(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Handle a model_information_updating message."""
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")
        peer = self.ctx.peers.get(source)
        if peer is None:
            logger.warning(self.ctx.address, f"Peer state not found for {source}")
            return
        try:
            base_model = await self.ctx.learner.aget_model()
            model = base_model.build_copy(
                params=weights,
                num_samples=num_samples,
                contributors=list(contributors),
            )
            peer.model = model
            logger.info(self.ctx.address, "Model received.")
        except DecodingParamsError:
            logger.error(self.ctx.address, "Error decoding parameters.")
        except ModelNotMatchingError:
            logger.error(self.ctx.address, "Models not matching.")
        except Exception as e:
            logger.error(self.ctx.address, f"Unknown error adding model: {e}")

    @on_message("push_sum_weight_information_updating")
    async def handle_push_sum_weight(self, source: str, round: int, *args) -> None:
        """Handle a push_sum_weight_information_updating message."""
        if not args:
            raise ValueError("Push-sum weight is required")
        push_sum_weight = float(args[0])
        peer = self.ctx.peers.get(source)
        if peer is None:
            logger.warning(self.ctx.address, f"Peer state not found for {source}")
            return
        try:
            peer.push_sum_weight = push_sum_weight
            logger.debug(self.ctx.address, f"{source} push-sum weight updated to {push_sum_weight}")
        except Exception as e:
            logger.error(self.ctx.address, f"Error saving push-sum weight from {source}: {e}")

    @on_message("pre_send_model_training")
    async def handle_pre_send_model_training(self, source: str, round: int, *args) -> str:
        """Handle a pre_send_model_training request by checking if the model should be accepted."""
        if not args:
            return "false"
        weight_command = args[0]
        logger.debug(self.ctx.address, f"pre_send_model from {source}: weight_command={weight_command}")
        contributors = list(args[1:]) if len(args) > 1 else []

        existing: set[str] = set()
        for p in self.ctx.peers.values():
            if p.model:
                existing.update(p.model.get_contributors())

        accepted = should_accept_model(
            weight_command=weight_command,
            contributors=contributors,
            round=round,
            local_round=self.ctx.experiment.round,
            existing_contributors=existing,
        )
        return "true" if accepted else "false"


def _windowed_avg_loss(losses: dict[int, float], t_hat: int, tau: int) -> float:
    """Average loss over rounds [t̂-τ, t̂] (Eq. 39 from AsyDFL paper)."""
    values = [losses[r] for r in range(t_hat - tau, t_hat + 1) if r in losses]
    return sum(values) / len(values) if values else 0.0


def compute_priority(
    ti: int,
    tp_ij: int,
    tj: int,
    tl_ji: int,
    f_ti: float,
    f_tj: float,
    dmax: int,
) -> float:
    """
    Compute neighbor priority p(b_ij) based on Equation 40.

    Combines communication staleness with training loss divergence
    to decide which neighbors should receive model updates.

    Args:
        ti: Index of the local iteration on node i.
        tp_ij: Index of the local iteration when node i pushes model to node j.
        tj: Index of the local iteration on node j.
        tl_ji: Index of the local iteration when node j updates with model from node i.
        f_ti: Training loss or function value at node i.
        f_tj: Training loss or function value at node j.
        dmax: Maximum communication frequency bound.

    Returns:
        The computed priority p(b_ij).

    """
    if dmax <= 0:
        raise ValueError("dmax must be positive")

    dij = min(abs((ti - tp_ij) - (tj - tl_ji)) / dmax, 1.0)
    try:
        loss_term = math.exp(abs(f_ti - f_tj)) / math.exp(1)
    except OverflowError:
        loss_term = float("inf")
    return dij + (1 - dij) * loss_term
