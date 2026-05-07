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

"""Tests for AsyncDFL workflow (new engine)."""

import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.workflow.async_dfl.context import AsyncDFLContext, AsyncPeerState
from p2pfl.workflow.async_dfl.stages.setup import SetupStage
from p2pfl.workflow.async_dfl.stages.training_round import TrainingRoundStage, _windowed_sum_loss, compute_priority
from p2pfl.workflow.async_dfl.workflow import AsyncDFL
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.factory import create_workflow
from p2pfl.workflow.validation import validate


class TestAsyncDFLCreation:
    """Tests for AsyncDFL workflow creation and initialization."""

    def test_initial_stage_is_setup(self):
        """Test that initial_stage is derived from the first stage."""
        wf = AsyncDFL()
        assert wf.initial_stage == wf.get_stages()[0].name

    def test_stages_map_has_all_stages(self):
        """Test that get_stages returns all expected stages."""
        wf = AsyncDFL()
        stages = wf.get_stages()
        assert {s.name for s in stages} == {"setup", "training_round", "finish"}

    def test_factory_creates_async(self):
        """Test that factory creates AsyncDFL for ASYNC type."""
        wf = create_workflow("async")
        assert isinstance(wf, AsyncDFL)


class TestAsyncDFLValidation:
    """Tests for AsyncDFL graph validation."""

    def test_graph_passes_validation(self):
        """AsyncDFL stage graph has no validation errors."""
        wf = AsyncDFL()
        result = validate(wf)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_transitions_match_expected_stages(self):
        """setup→training_round, training_round→{training_round,finish}, finish→None."""
        wf = AsyncDFL()
        result = validate(wf)
        transitions = {k: v.targets for k, v in result.transitions.items()}
        assert "training_round" in transitions["setup"]
        assert "training_round" in transitions["training_round"]
        assert "finish" in transitions["training_round"]
        assert transitions["finish"] == {None}


class TestAsyncDFLDeclaredMessages:
    """Tests for message declaration (before run)."""

    def test_declared_messages_contain_all(self):
        """Test that get_messages returns all expected messages."""
        wf = AsyncDFL()
        msgs = wf.get_messages()
        expected = {
            "node_initialized",
            "loss_information_updating",
            "index_information_updating",
            "model_information_updating",
            "push_sum_weight_information_updating",
            "pre_send_model_training",
        }
        assert set(msgs.keys()) == expected

    def test_weights_messages_flagged(self):
        """Test that weight messages are properly flagged."""
        wf = AsyncDFL()
        msgs = wf.get_messages()
        assert msgs["model_information_updating"].is_weights is True
        assert msgs["node_initialized"].is_weights is False
        assert msgs["loss_information_updating"].is_weights is False
        assert msgs["push_sum_weight_information_updating"].is_weights is False

    def test_during_filters_set(self):
        """
        Test that during filters are set on all handlers.

        Pre-compose, handlers without explicit ``during`` have ``during=None``
        (the owning-stage default is applied during ``_compose``).
        """
        wf = AsyncDFL()
        msgs = wf.get_messages()
        # All AsyncDFL handlers use @on_message without during= → None pre-compose
        assert msgs["node_initialized"].during is None
        assert msgs["loss_information_updating"].during is None
        assert msgs["index_information_updating"].during is None
        assert msgs["model_information_updating"].during is None
        assert msgs["push_sum_weight_information_updating"].during is None
        assert msgs["pre_send_model_training"].during is None


class TestAsyncPeerState:
    """Tests for AsyncPeerState operations."""

    def test_default_values(self):
        """Test that AsyncPeerState has correct defaults."""
        peer = AsyncPeerState()
        assert peer.round_number == 0
        assert peer.push_sum_weight == 1.0
        assert peer.model is None
        assert peer.losses == {}
        assert peer.push_time == 0
        assert peer.mixing_weight == 1.0
        assert peer.p2p_updating_idx == 0

    def test_add_loss(self):
        """Test add_loss sets loss at round index in dict."""
        peer = AsyncPeerState()
        peer.add_loss(0, 0.5)
        assert peer.losses == {0: 0.5}
        peer.add_loss(3, 0.2)
        assert peer.losses == {0: 0.5, 3: 0.2}

    def test_add_loss_overwrites(self):
        """Test add_loss overwrites existing value."""
        peer = AsyncPeerState()
        peer.add_loss(0, 0.5)
        peer.add_loss(0, 0.9)
        assert peer.losses == {0: 0.9}

    def test_reset_round(self):
        """Test reset_round clears model."""
        peer = AsyncPeerState()
        peer.model = MagicMock()
        peer.reset_round()
        assert peer.model is None


class TestComputePriority:
    """Tests for the compute_priority function."""

    def test_returns_nonnegative_float(self):
        """compute_priority returns a non-negative float for typical inputs."""
        p = compute_priority(ti=10, tp_ij=5, tj=8, tl_ji=3, f_ti=0.5, f_tj=0.5, dmax=5)
        assert isinstance(p, float)
        assert p >= 0

    def test_zero_loss_difference(self):
        """Test priority with zero loss difference."""
        p = compute_priority(ti=0, tp_ij=0, tj=0, tl_ji=0, f_ti=1.0, f_tj=1.0, dmax=5)
        # dij = 0, loss_term = exp(0)/exp(1) ≈ 0.368
        assert abs(p - 0.368) < 0.01

    def test_high_staleness(self):
        """Test priority with high staleness."""
        p = compute_priority(ti=10, tp_ij=0, tj=0, tl_ji=0, f_ti=0.5, f_tj=0.5, dmax=5)
        # dij = min(10/5, 1.0) = 1.0, so priority = 1.0
        assert abs(p - 1.0) < 0.01

    def test_dmax_must_be_positive(self):
        """Test that dmax <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="dmax must be positive"):
            compute_priority(ti=0, tp_ij=0, tj=0, tl_ji=0, f_ti=0.0, f_tj=0.0, dmax=0)

    def test_overflow_handled(self):
        """Test that large loss differences don't crash."""
        p = compute_priority(ti=0, tp_ij=0, tj=0, tl_ji=0, f_ti=0.0, f_tj=1000.0, dmax=5)
        assert p == float("inf")


class TestTrainingRoundStageConditions:
    """Tests for TrainingRoundStage condition helpers."""

    def test_select_neighbors_top_3(self):
        """Test _select_neighbors picks top 3 by priority."""
        priorities = [("a", 1.0, False), ("b", 3.0, False), ("c", 2.0, False), ("d", 0.5, False), ("e", 4.0, False)]
        result = TrainingRoundStage._select_neighbors(priorities)
        assert result == ["e", "b", "c"]

    def test_select_neighbors_fewer_than_3(self):
        """Test _select_neighbors with fewer than 3 neighbors."""
        priorities = [("a", 1.0, False), ("b", 2.0, False)]
        result = TrainingRoundStage._select_neighbors(priorities)
        assert result == ["b", "a"]

    def test_select_neighbors_mandatory_included(self):
        """Mandatory neighbors (d_{i,j} >= d_max) are always included."""
        priorities = [("a", 0.1, True), ("b", 3.0, False), ("c", 2.0, False), ("d", 0.5, False), ("e", 4.0, False)]
        result = TrainingRoundStage._select_neighbors(priorities, top_k=3)
        assert "a" in result
        assert len(result) == 3

    def test_select_neighbors_mandatory_exceeds_top_k(self):
        """When mandatory count exceeds top_k, all mandatory are still included."""
        priorities = [("a", 1.0, True), ("b", 1.0, True), ("c", 1.0, True), ("d", 1.0, True), ("e", 0.5, False)]
        result = TrainingRoundStage._select_neighbors(priorities, top_k=2)
        assert set(result) == {"a", "b", "c", "d"}


class TestAsyncDFLContext:
    """Tests for AsyncDFLContext creation."""

    def test_create_context(self):
        """Test that create_context builds a proper AsyncDFLContext."""
        wf = AsyncDFL()
        ctx = wf.create_context(
            address="test",
            learner=MagicMock(),
            aggregator=MagicMock(),
            cp=MagicMock(),
            generator=MagicMock(),
            experiment=Experiment.create(exp_name="test", total_rounds=5, tau=3),
        )
        assert isinstance(ctx, AsyncDFLContext)
        assert ctx.address == "test"
        assert ctx.experiment.tau == 3
        assert ctx.peers == {}
        assert ctx.candidates == []

    def test_create_context_default_tau(self):
        """Test that validate_experiment defaults tau to 2."""
        wf = AsyncDFL()
        ctx = wf.create_context(
            address="test",
            learner=MagicMock(),
            aggregator=MagicMock(),
            cp=MagicMock(),
            generator=MagicMock(),
            experiment=Experiment("test", total_rounds=5),
        )
        wf.validate_experiment(ctx)
        assert ctx.experiment.tau == 2


###
# Helpers for stage-level tests
###


def _make_ctx(
    address: str = "node-1:5000",
    total_rounds: int = 5,
    tau: int = 2,
    dmax: int = 5,
    top_k_neighbors: int = 3,
    round: int = 0,
    neighbors: list[str] | None = None,
) -> AsyncDFLContext:
    """Build an AsyncDFLContext with fully-mocked collaborators."""
    if neighbors is None:
        neighbors = []

    learner = MagicMock()
    learner.train_on_batch = AsyncMock()
    learner.evaluate = AsyncMock(return_value={})
    model = MagicMock()
    model.last_training_loss = 0.5
    model.encode_parameters.return_value = b"fake"
    model.get_contributors.return_value = [address]
    model.get_num_samples.return_value = 100
    model.get_info.return_value = {}
    model.build_copy.return_value = MagicMock()
    learner.get_model.return_value = model
    learner.set_model = MagicMock()
    learner.set_epochs = MagicMock()

    cp = MagicMock()
    cp.get_neighbors.return_value = neighbors
    cp.broadcast_gossip = AsyncMock()
    cp.send = AsyncMock(return_value="true")
    cp.build_msg.return_value = "msg"
    cp.build_weights.return_value = "weights_payload"

    aggregator = MagicMock()
    agg_model = MagicMock()
    agg_model.get_info.return_value = {"push_sum_weight": 0.7}
    aggregator.aggregate.return_value = agg_model

    exp = Experiment.create(
        exp_name="test",
        total_rounds=total_rounds,
        tau=tau,
        dmax=dmax,
        top_k_neighbors=top_k_neighbors,
    )
    exp.round = round

    return AsyncDFLContext(
        address=address,
        learner=learner,
        aggregator=aggregator,
        cp=cp,
        generator=random.Random(42),
        experiment=exp,
    )


def _attach_stage(stage, ctx):
    """Mimic what the workflow engine does: inject ctx into stage."""
    stage.ctx = ctx
    return stage


###
# _windowed_sum_loss
###


class TestWindowedSumLoss:
    """Tests for the _windowed_sum_loss helper."""

    def test_exact_window(self):
        """Sum over a full window of rounds (Eq. 39)."""
        losses = {0: 1.0, 1: 2.0, 2: 3.0}
        assert _windowed_sum_loss(losses, t_hat=2, tau=2) == 6.0

    def test_partial_window(self):
        """Missing rounds are skipped, sum is over available values only."""
        losses = {2: 4.0}
        assert _windowed_sum_loss(losses, t_hat=2, tau=2) == 4.0

    def test_empty_losses(self):
        """Returns 0.0 when no matching rounds exist."""
        assert _windowed_sum_loss({}, t_hat=5, tau=3) == 0.0

    def test_single_round(self):
        """tau=0 means window is [t_hat, t_hat]."""
        losses = {3: 7.0}
        assert _windowed_sum_loss(losses, t_hat=3, tau=0) == 7.0


###
# TrainingRoundStage.run
###


class TestTrainingRoundRun:
    """Tests for the main training round loop."""

    @pytest.mark.asyncio
    async def test_run_returns_training_round_when_not_complete(self):
        """run() returns 'training_round' while experiment is not done."""
        ctx = _make_ctx(total_rounds=5, round=0, tau=10)
        stage = _attach_stage(TrainingRoundStage(), ctx)

        result = await stage.run()

        assert result == "training_round"
        assert ctx.experiment.round == 1
        ctx.learner.train_on_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_returns_finish_when_complete(self):
        """run() returns 'finish' when experiment reaches total_rounds."""
        ctx = _make_ctx(total_rounds=1, round=0, tau=10)
        stage = _attach_stage(TrainingRoundStage(), ctx)

        result = await stage.run()

        assert result == "finish"
        assert ctx.experiment.round == 1

    @pytest.mark.asyncio
    async def test_run_increments_round_without_clearing_models(self):
        """run() increments round counter; models are NOT cleared outside τ boundaries."""
        ctx = _make_ctx(total_rounds=5, round=2, tau=100)
        remote_model = MagicMock()
        peer_a = AsyncPeerState()
        peer_b = AsyncPeerState(model=remote_model)
        ctx.peers = {"node-1:5000": peer_a, "node-2:5000": peer_b}

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage.run()

        assert ctx.experiment.round == 3
        assert peer_b.model is remote_model

    @pytest.mark.asyncio
    async def test_run_stores_trained_model_on_local_peer(self):
        """After training, the local peer's model is set and persists (no per-round clear)."""
        ctx = _make_ctx(total_rounds=5, round=0, tau=100)
        ctx.peers[ctx.address] = AsyncPeerState()
        trained_model = MagicMock()
        ctx.learner.get_model.return_value = trained_model

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage.run()

        assert ctx.peers[ctx.address].model is trained_model

    @pytest.mark.asyncio
    async def test_run_handles_missing_local_peer(self):
        """run() does not crash if local peer state is missing."""
        ctx = _make_ctx(total_rounds=5, round=0, tau=100)
        ctx.peers = {}  # no local peer
        stage = _attach_stage(TrainingRoundStage(), ctx)

        result = await stage.run()
        assert result == "training_round"

    @pytest.mark.asyncio
    async def test_run_triggers_network_update_at_tau_boundary(self):
        """Network update runs when round > 0 and round % tau == 0."""
        ctx = _make_ctx(
            total_rounds=10,
            round=1,  # after increment it will be 2; but network_update happens before increment
            tau=2,
            neighbors=["node-2:5000"],
        )
        # round=1, tau=2 -> round(1) % tau(2) != 0, no network update
        # We need round=2 to trigger: round > 0 and round % tau == 0
        ctx.experiment.round = 2
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.peers["node-2:5000"] = AsyncPeerState()

        stage = _attach_stage(TrainingRoundStage(), ctx)
        with patch.object(stage, "_network_update", new_callable=AsyncMock) as mock_nu:
            await stage.run()
            mock_nu.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_skips_network_update_at_round_zero(self):
        """Network update is skipped at round 0 even if tau divides 0."""
        ctx = _make_ctx(total_rounds=5, round=0, tau=1)
        stage = _attach_stage(TrainingRoundStage(), ctx)
        with patch.object(stage, "_network_update", new_callable=AsyncMock) as mock_nu:
            await stage.run()
            mock_nu.assert_not_awaited()


###
# TrainingRoundStage._debias_model
###


class TestDebiasModel:
    """Tests for the debiasing phase."""

    def test_sets_push_sum_weight_on_model(self):
        """Debias sets push_sum_weight when model supports it."""
        ctx = _make_ctx()
        ctx.peers[ctx.address] = AsyncPeerState(push_sum_weight=0.42)
        model = ctx.learner.get_model()
        model.set_push_sum_weight = MagicMock()

        stage = _attach_stage(TrainingRoundStage(), ctx)
        stage._debias_model(ctx)

        model.set_push_sum_weight.assert_called_once_with(0.42)

    def test_skips_if_no_local_peer(self):
        """Debias does not crash if local peer is missing."""
        ctx = _make_ctx()
        ctx.peers = {}
        stage = _attach_stage(TrainingRoundStage(), ctx)
        stage._debias_model(ctx)  # should not raise

    def test_skips_if_model_lacks_push_sum(self):
        """Debias is a no-op when model does not support set_push_sum_weight."""
        ctx = _make_ctx()
        ctx.peers[ctx.address] = AsyncPeerState()
        model = ctx.learner.get_model()
        del model.set_push_sum_weight  # remove the attribute

        stage = _attach_stage(TrainingRoundStage(), ctx)
        stage._debias_model(ctx)  # should not raise


###
# TrainingRoundStage._broadcast_loss
###


class TestBroadcastLoss:
    """Tests for loss broadcasting phase."""

    @pytest.mark.asyncio
    async def test_broadcasts_loss_to_peers(self):
        """Loss is recorded locally and broadcast via gossip."""
        ctx = _make_ctx()
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.experiment.round = 3
        model = ctx.learner.get_model()
        model.last_training_loss = 0.25

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._broadcast_loss(ctx)

        assert ctx.peers[ctx.address].losses[3] == 0.25
        ctx.cp.broadcast_gossip.assert_awaited_once()
        ctx.cp.build_msg.assert_called_with("loss_information_updating", ["0.25"], round=3)

    @pytest.mark.asyncio
    async def test_broadcast_loss_handles_exception(self):
        """broadcast_loss does not crash if gossip fails."""
        ctx = _make_ctx()
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.cp.broadcast_gossip = AsyncMock(side_effect=RuntimeError("network error"))

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._broadcast_loss(ctx)  # should not raise

    @pytest.mark.asyncio
    async def test_broadcast_loss_missing_local_peer(self):
        """broadcast_loss does not crash when local peer is absent."""
        ctx = _make_ctx()
        ctx.peers = {}
        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._broadcast_loss(ctx)  # should not raise


###
# TrainingRoundStage._compute_priorities
###


class TestComputePriorities:
    """Tests for priority computation across neighbors."""

    def test_computes_priority_for_each_neighbor(self):
        """Returns a (name, priority, mandatory) tuple for each direct neighbor with peer state."""
        ctx = _make_ctx(neighbors=["n1", "n2"])
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.peers["n1"] = AsyncPeerState(round_number=1, push_time=0, p2p_updating_idx=0)
        ctx.peers["n2"] = AsyncPeerState(round_number=2, push_time=1, p2p_updating_idx=1)
        ctx.experiment.round = 3

        stage = _attach_stage(TrainingRoundStage(), ctx)
        result = stage._compute_priorities(ctx)

        assert len(result) == 2
        names = {n for n, _, _ in result}
        assert names == {"n1", "n2"}
        assert all(isinstance(p, float) for _, p, _ in result)
        assert all(isinstance(m, bool) for _, _, m in result)

    def test_skips_neighbor_without_peer_state(self):
        """Neighbors without an entry in ctx.peers are skipped."""
        ctx = _make_ctx(neighbors=["n1", "n2"])
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.peers["n1"] = AsyncPeerState()
        # n2 has no peer state

        stage = _attach_stage(TrainingRoundStage(), ctx)
        result = stage._compute_priorities(ctx)

        assert len(result) == 1
        assert result[0][0] == "n1"

    def test_uses_windowed_loss_correctly(self):
        """Priority uses windowed sum loss over [t_hat-tau, t_hat]."""
        ctx = _make_ctx(neighbors=["n1"], tau=2, dmax=10)
        ctx.experiment.round = 5
        local_peer = AsyncPeerState()
        local_peer.losses = {3: 0.3, 4: 0.4, 5: 0.5}
        ctx.peers[ctx.address] = local_peer
        neighbor_peer = AsyncPeerState(round_number=4)
        neighbor_peer.losses = {2: 0.2, 3: 0.3, 4: 0.4}
        ctx.peers["n1"] = neighbor_peer

        stage = _attach_stage(TrainingRoundStage(), ctx)
        result = stage._compute_priorities(ctx)

        assert len(result) == 1
        _, priority, _ = result[0]
        assert isinstance(priority, float)
        assert priority > 0

    def test_mandatory_flag_when_stale(self):
        """Neighbor is mandatory when d_{i,j} >= d_max."""
        ctx = _make_ctx(neighbors=["n1"], dmax=3)
        ctx.experiment.round = 10
        ctx.peers[ctx.address] = AsyncPeerState()
        # d_{i,j} = |(10 - 0) - (0 - 0)| = 10 >= 3
        ctx.peers["n1"] = AsyncPeerState(push_time=0, round_number=0, p2p_updating_idx=0)

        stage = _attach_stage(TrainingRoundStage(), ctx)
        result = stage._compute_priorities(ctx)

        assert len(result) == 1
        _, _, is_mandatory = result[0]
        assert is_mandatory is True

    def test_not_mandatory_when_fresh(self):
        """Neighbor is not mandatory when d_{i,j} < d_max."""
        ctx = _make_ctx(neighbors=["n1"], dmax=10)
        ctx.experiment.round = 5
        ctx.peers[ctx.address] = AsyncPeerState()
        # d_{i,j} = |(5 - 4) - (4 - 3)| = 0 < 10
        ctx.peers["n1"] = AsyncPeerState(push_time=4, round_number=4, p2p_updating_idx=3)

        stage = _attach_stage(TrainingRoundStage(), ctx)
        result = stage._compute_priorities(ctx)

        assert len(result) == 1
        _, _, is_mandatory = result[0]
        assert is_mandatory is False


###
# TrainingRoundStage._network_update (integration of gossip + aggregate)
###


class TestNetworkUpdate:
    """Tests for the full network update flow."""

    @pytest.mark.asyncio
    async def test_network_update_selects_neighbors_and_gossips(self):
        """network_update computes priorities, selects neighbors, gossips, aggregates."""
        ctx = _make_ctx(neighbors=["n1", "n2"], tau=2, top_k_neighbors=2, dmax=5)
        ctx.experiment.round = 2
        # Give peers models so aggregation has something to work with
        ctx.peers[ctx.address] = AsyncPeerState(model=MagicMock())
        ctx.peers["n1"] = AsyncPeerState(model=MagicMock())
        ctx.peers["n2"] = AsyncPeerState(model=MagicMock())

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._network_update(ctx)

        # Candidates should be set
        assert len(ctx.candidates) <= 2
        # Aggregator should have been called
        ctx.aggregator.aggregate.assert_called_once()
        ctx.learner.set_model.assert_called_once()

        # Remote peer models cleared after aggregation (Alg. 4 line 18)
        assert ctx.peers["n1"].model is None
        assert ctx.peers["n2"].model is None
        # Local peer model is NOT cleared
        assert ctx.peers[ctx.address].model is not None

    @pytest.mark.asyncio
    async def test_network_update_skips_gossip_when_no_neighbors(self):
        """network_update still runs aggregation even with no neighbors."""
        ctx = _make_ctx(neighbors=[], tau=2)
        ctx.experiment.round = 2
        ctx.peers[ctx.address] = AsyncPeerState()

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._network_update(ctx)

        assert ctx.candidates == []


###
# TrainingRoundStage._gossip_model
###


class TestGossipModel:
    """Tests for model gossiping to selected neighbors."""

    @pytest.mark.asyncio
    async def test_sends_model_to_each_candidate(self):
        """Model is sent to each candidate via ModelGate."""
        ctx = _make_ctx(neighbors=["n1", "n2"])
        ctx.candidates = ["n1", "n2"]
        ctx.experiment.round = 4
        ctx.peers[ctx.address] = AsyncPeerState(push_sum_weight=0.5)
        ctx.peers["n1"] = AsyncPeerState()
        ctx.peers["n2"] = AsyncPeerState()
        # cp.send returns "true" so gate accepts
        ctx.cp.send = AsyncMock(return_value="true")

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._gossip_model(ctx)

        # push_time should be updated for neighbors
        assert ctx.peers["n1"].push_time == 4
        assert ctx.peers["n2"].push_time == 4

    @pytest.mark.asyncio
    async def test_skips_sending_when_gate_rejects(self):
        """When gate returns 'false', model is not sent and push_time is not updated."""
        ctx = _make_ctx(neighbors=["n1"])
        ctx.candidates = ["n1"]
        ctx.experiment.round = 3
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.peers["n1"] = AsyncPeerState(push_time=0)
        ctx.cp.send = AsyncMock(return_value="false")

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._gossip_model(ctx)

        # push_time should NOT be updated
        assert ctx.peers["n1"].push_time == 0

    @pytest.mark.asyncio
    async def test_gossip_handles_missing_peer_state(self):
        """Gossiping to a candidate with no peer state logs a warning but does not crash."""
        ctx = _make_ctx(neighbors=["n1"])
        ctx.candidates = ["n1"]
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.cp.send = AsyncMock(return_value="true")
        # n1 has no peer state entry

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._gossip_model(ctx)  # should not raise


###
# TrainingRoundStage._send_push_sum_weight
###


class TestSendPushSumWeight:
    """Tests for sending push-sum weight to a neighbor."""

    @pytest.mark.asyncio
    async def test_sends_push_sum_weight(self):
        """Sends local push-sum weight via cp.send."""
        ctx = _make_ctx()
        ctx.peers[ctx.address] = AsyncPeerState(push_sum_weight=0.75)
        ctx.experiment.round = 5

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._send_push_sum_weight(ctx, "neighbor-1")

        ctx.cp.send.assert_awaited_once()
        ctx.cp.build_msg.assert_called_with("push_sum_weight_information_updating", [0.75], round=5)

    @pytest.mark.asyncio
    async def test_uses_default_weight_if_no_self_peer(self):
        """Falls back to 1.0 if local peer state is missing."""
        ctx = _make_ctx()
        ctx.peers = {}
        ctx.experiment.round = 1

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._send_push_sum_weight(ctx, "neighbor-1")

        ctx.cp.build_msg.assert_called_with("push_sum_weight_information_updating", [1.0], round=1)

    @pytest.mark.asyncio
    async def test_handles_send_failure(self):
        """Does not crash if cp.send raises."""
        ctx = _make_ctx()
        ctx.peers[ctx.address] = AsyncPeerState()
        ctx.cp.send = AsyncMock(side_effect=RuntimeError("down"))

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._send_push_sum_weight(ctx, "neighbor-1")  # should not raise


###
# TrainingRoundStage._aggregate
###


class TestAggregate:
    """Tests for the aggregation phase."""

    @pytest.mark.asyncio
    async def test_aggregates_all_peer_models(self):
        """Aggregate collects models from all peers (including local) and calls aggregator."""
        ctx = _make_ctx(neighbors=["n1"])
        ctx.experiment.round = 3
        local_model = MagicMock()
        remote_model = MagicMock()
        ctx.peers[ctx.address] = AsyncPeerState(model=local_model, mixing_weight=0.5, push_sum_weight=0.8)
        ctx.peers["n1"] = AsyncPeerState(model=remote_model, mixing_weight=0.5, push_sum_weight=0.2)

        agg_result = MagicMock()
        agg_result.get_info.return_value = {"push_sum_weight": 0.6}
        ctx.aggregator.aggregate.return_value = agg_result

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._aggregate(ctx)

        # Both models should have mixing/push_sum info added
        local_model.add_info.assert_any_call("mixing_weight", 0.5)
        local_model.add_info.assert_any_call("push_sum_weight", 0.8)
        remote_model.add_info.assert_any_call("mixing_weight", 0.5)
        remote_model.add_info.assert_any_call("push_sum_weight", 0.2)

        # Aggregator called with both models
        ctx.aggregator.aggregate.assert_called_once()
        models_arg = ctx.aggregator.aggregate.call_args[0][0]
        assert len(models_arg) == 2

        # Learner gets the aggregated model
        ctx.learner.set_model.assert_called_once_with(agg_result)

        # Local push_sum_weight updated from aggregation result
        assert ctx.peers[ctx.address].push_sum_weight == 0.6

    @pytest.mark.asyncio
    async def test_aggregate_does_not_set_p2p_updating_idx_locally(self):
        """p2p_updating_idx is updated via messages (Alg. 3), not locally during aggregate."""
        ctx = _make_ctx(neighbors=["n1"])
        ctx.experiment.round = 7
        ctx.peers[ctx.address] = AsyncPeerState(model=MagicMock())
        ctx.peers["n1"] = AsyncPeerState(model=MagicMock(), p2p_updating_idx=0)

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._aggregate(ctx)

        assert ctx.peers["n1"].p2p_updating_idx == 0

    @pytest.mark.asyncio
    async def test_aggregate_skips_peers_without_models(self):
        """Peers with model=None are excluded from aggregation."""
        ctx = _make_ctx(neighbors=["n1"])
        ctx.peers[ctx.address] = AsyncPeerState(model=MagicMock())
        ctx.peers["n1"] = AsyncPeerState(model=None)

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._aggregate(ctx)

        models_arg = ctx.aggregator.aggregate.call_args[0][0]
        assert len(models_arg) == 1

    @pytest.mark.asyncio
    async def test_aggregate_no_models_skips_aggregation(self):
        """When no peers have models, aggregator is not called."""
        ctx = _make_ctx()
        ctx.peers[ctx.address] = AsyncPeerState(model=None)

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._aggregate(ctx)

        ctx.aggregator.aggregate.assert_not_called()

    @pytest.mark.asyncio
    async def test_aggregate_handles_send_failure_to_remote(self):
        """Aggregate does not crash if sending index_information_updating to a remote peer fails."""
        ctx = _make_ctx(neighbors=["n1"])
        ctx.experiment.round = 2
        ctx.peers[ctx.address] = AsyncPeerState(model=MagicMock())
        ctx.peers["n1"] = AsyncPeerState(model=MagicMock())
        ctx.cp.send = AsyncMock(side_effect=RuntimeError("fail"))

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage._aggregate(ctx)  # should not raise

        # Aggregator should still have been called
        ctx.aggregator.aggregate.assert_called_once()


###
# TrainingRoundStage message handlers
###


class TestTrainingRoundMessageHandlers:
    """Tests for @on_message handlers on TrainingRoundStage."""

    @pytest.mark.asyncio
    async def test_handle_loss_information_records_loss(self):
        """handle_loss_information stores loss and updates round_number (t_j)."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState()
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_loss_information("sender", 3, "0.42")

        assert ctx.peers["sender"].losses[3] == 0.42
        assert ctx.peers["sender"].round_number == 3

    @pytest.mark.asyncio
    async def test_handle_loss_information_round_number_monotonic(self):
        """round_number only increases (handles out-of-order messages)."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState(round_number=5)
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_loss_information("sender", 3, "0.1")

        assert ctx.peers["sender"].round_number == 5

    @pytest.mark.asyncio
    async def test_handle_loss_information_missing_args(self):
        """handle_loss_information raises ValueError when no loss provided."""
        ctx = _make_ctx()
        stage = _attach_stage(TrainingRoundStage(), ctx)

        with pytest.raises(ValueError, match="Loss value is required"):
            await stage.handle_loss_information("sender", 3)

    @pytest.mark.asyncio
    async def test_handle_loss_information_unknown_peer(self):
        """handle_loss_information returns without error for unknown peers."""
        ctx = _make_ctx()
        ctx.peers = {}
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_loss_information("unknown", 1, "0.5")  # should not raise

    @pytest.mark.asyncio
    async def test_handle_index_information_updates_p2p_updating_idx(self):
        """handle_index_information updates peer's p2p_updating_idx (Alg. 3: t^l_{j,i})."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState(p2p_updating_idx=0)
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_index_information("sender", 5)

        assert ctx.peers["sender"].p2p_updating_idx == 5

    @pytest.mark.asyncio
    async def test_handle_index_information_monotonic(self):
        """p2p_updating_idx only increases (handles out-of-order messages)."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState(p2p_updating_idx=7)
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_index_information("sender", 3)

        assert ctx.peers["sender"].p2p_updating_idx == 7

    @pytest.mark.asyncio
    async def test_handle_index_information_unknown_peer(self):
        """handle_index_information handles unknown peer gracefully."""
        ctx = _make_ctx()
        ctx.peers = {}
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_index_information("unknown", 5)  # should not raise

    @pytest.mark.asyncio
    async def test_handle_model_information_stores_model(self):
        """handle_model_information builds and stores model copy in peer state."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState()
        built_model = MagicMock()
        ctx.learner.get_model().build_copy.return_value = built_model

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage.handle_model_information(
            source="sender",
            round=2,
            weights=b"encoded",
            contributors=["sender"],
            num_samples=50,
        )

        assert ctx.peers["sender"].model is built_model

    @pytest.mark.asyncio
    async def test_handle_model_information_missing_contributors(self):
        """Raises ValueError when contributors or num_samples is None."""
        ctx = _make_ctx()
        stage = _attach_stage(TrainingRoundStage(), ctx)

        with pytest.raises(ValueError, match="Contributors and num_samples are required"):
            await stage.handle_model_information("sender", 2, b"data", None, None)

    @pytest.mark.asyncio
    async def test_handle_model_information_unknown_peer(self):
        """handle_model_information returns gracefully for unknown peer."""
        ctx = _make_ctx()
        ctx.peers = {}
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_model_information("unknown", 2, b"data", ["a"], 10)

    @pytest.mark.asyncio
    async def test_handle_model_information_decoding_error(self):
        """handle_model_information catches DecodingParamsError."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState()
        ctx.learner.get_model().build_copy.side_effect = DecodingParamsError("bad params")

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage.handle_model_information("sender", 2, b"bad", ["sender"], 10)

        assert ctx.peers["sender"].model is None

    @pytest.mark.asyncio
    async def test_handle_model_information_model_not_matching(self):
        """handle_model_information catches ModelNotMatchingError."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState()
        ctx.learner.get_model().build_copy.side_effect = ModelNotMatchingError("mismatch")

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage.handle_model_information("sender", 2, b"bad", ["sender"], 10)

        assert ctx.peers["sender"].model is None

    @pytest.mark.asyncio
    async def test_handle_model_information_generic_error(self):
        """handle_model_information catches generic exceptions."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState()
        ctx.learner.get_model().build_copy.side_effect = RuntimeError("unexpected")

        stage = _attach_stage(TrainingRoundStage(), ctx)
        await stage.handle_model_information("sender", 2, b"bad", ["sender"], 10)

        assert ctx.peers["sender"].model is None

    @pytest.mark.asyncio
    async def test_handle_push_sum_weight_updates_peer(self):
        """handle_push_sum_weight stores the weight in peer state."""
        ctx = _make_ctx()
        ctx.peers["sender"] = AsyncPeerState(push_sum_weight=1.0)
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_push_sum_weight("sender", 3, "0.55")

        assert ctx.peers["sender"].push_sum_weight == 0.55

    @pytest.mark.asyncio
    async def test_handle_push_sum_weight_missing_args(self):
        """Raises ValueError when no weight provided."""
        ctx = _make_ctx()
        stage = _attach_stage(TrainingRoundStage(), ctx)

        with pytest.raises(ValueError, match="Push-sum weight is required"):
            await stage.handle_push_sum_weight("sender", 1)

    @pytest.mark.asyncio
    async def test_handle_push_sum_weight_unknown_peer(self):
        """Handles unknown peer gracefully."""
        ctx = _make_ctx()
        ctx.peers = {}
        stage = _attach_stage(TrainingRoundStage(), ctx)

        await stage.handle_push_sum_weight("unknown", 1, "0.5")

    @pytest.mark.asyncio
    async def test_handle_pre_send_model_training_accepts(self):
        """Returns 'true' when sender round >= local round."""
        ctx = _make_ctx()
        ctx.experiment.round = 5
        stage = _attach_stage(TrainingRoundStage(), ctx)

        result = await stage.handle_pre_send_model_training("sender", 5, "model_information_updating")
        assert result == "true"

        result = await stage.handle_pre_send_model_training("sender", 6, "model_information_updating")
        assert result == "true"

    @pytest.mark.asyncio
    async def test_handle_pre_send_model_training_rejects(self):
        """Returns 'false' when sender round < local round (stale)."""
        ctx = _make_ctx()
        ctx.experiment.round = 10
        stage = _attach_stage(TrainingRoundStage(), ctx)

        result = await stage.handle_pre_send_model_training("sender", 5, "model_information_updating")
        assert result == "false"

    @pytest.mark.asyncio
    async def test_handle_pre_send_model_training_no_args(self):
        """Returns 'false' when no args provided."""
        ctx = _make_ctx()
        stage = _attach_stage(TrainingRoundStage(), ctx)

        result = await stage.handle_pre_send_model_training("sender", 1)
        assert result == "false"


###
# SetupStage
###


class TestSetupStage:
    """Tests for the setup/synchronization stage."""

    @pytest.mark.asyncio
    async def test_run_returns_training_round(self):
        """SetupStage.run() returns 'training_round' after setup."""
        ctx = _make_ctx(neighbors=[])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        with patch("p2pfl.workflow.async_dfl.stages.setup.Settings") as mock_settings:
            mock_settings.training.SYNCHRONIZATION_TIMEOUT = 0.1
            result = await stage.run()

        assert result == "training_round"
        ctx.learner.set_epochs.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_creates_local_peer(self):
        """SetupStage creates a peer entry for the local node."""
        ctx = _make_ctx(neighbors=[])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        with patch("p2pfl.workflow.async_dfl.stages.setup.Settings") as mock_settings:
            mock_settings.training.SYNCHRONIZATION_TIMEOUT = 0.1
            await stage.run()

        assert ctx.address in ctx.peers

    @pytest.mark.asyncio
    async def test_run_sets_mixing_weights(self):
        """After setup, mixing weights are 1/(neighbors+1) for all peers."""
        ctx = _make_ctx(neighbors=["n1", "n2"])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        # Pre-create peers so that _all_nodes_started triggers
        ctx.peers["n1"] = AsyncPeerState()
        ctx.peers["n2"] = AsyncPeerState()

        with patch("p2pfl.workflow.async_dfl.stages.setup.Settings") as mock_settings:
            mock_settings.training.SYNCHRONIZATION_TIMEOUT = 0.1
            await stage.run()

        expected_weight = 1.0 / 3  # 2 neighbors + self
        assert abs(ctx.peers[ctx.address].mixing_weight - expected_weight) < 1e-9
        assert abs(ctx.peers["n1"].mixing_weight - expected_weight) < 1e-9
        assert abs(ctx.peers["n2"].mixing_weight - expected_weight) < 1e-9

    @pytest.mark.asyncio
    async def test_run_proceeds_on_timeout(self):
        """SetupStage proceeds even if not all nodes respond (timeout)."""
        ctx = _make_ctx(neighbors=["n1", "n2"])
        stage = SetupStage()
        _attach_stage(stage, ctx)
        # Don't pre-create any neighbor peers - they won't all be "ready"

        with patch("p2pfl.workflow.async_dfl.stages.setup.Settings") as mock_settings:
            mock_settings.training.SYNCHRONIZATION_TIMEOUT = 0.1
            result = await stage.run()

        assert result == "training_round"

    @pytest.mark.asyncio
    async def test_run_broadcasts_node_initialized(self):
        """SetupStage broadcasts node_initialized to discover peers."""
        # Need at least one neighbor so _all_nodes_started isn't immediately True
        # (with no neighbors, local peer alone triggers ready before broadcast loop)
        ctx = _make_ctx(neighbors=["n1"])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        with patch("p2pfl.workflow.async_dfl.stages.setup.Settings") as mock_settings:
            mock_settings.training.SYNCHRONIZATION_TIMEOUT = 0.2
            await stage.run()

        ctx.cp.broadcast_gossip.assert_called()
        ctx.cp.build_msg.assert_any_call("node_initialized")

    @pytest.mark.asyncio
    async def test_run_evaluates_before_training(self):
        """SetupStage calls evaluate_and_broadcast before returning."""
        ctx = _make_ctx(neighbors=[])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        with patch("p2pfl.workflow.async_dfl.stages.setup.Settings") as mock_settings:
            mock_settings.training.SYNCHRONIZATION_TIMEOUT = 0.1
            await stage.run()

        ctx.learner.evaluate.assert_awaited()

    @pytest.mark.asyncio
    async def test_create_peer_ignores_non_neighbor(self):
        """_create_peer ignores sources that are not neighbors or self."""
        ctx = _make_ctx(neighbors=["n1"])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        await stage._create_peer(ctx, source="stranger")
        assert "stranger" not in ctx.peers

    @pytest.mark.asyncio
    async def test_create_peer_idempotent(self):
        """_create_peer does not overwrite existing peer state."""
        ctx = _make_ctx(neighbors=["n1"])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        existing_peer = AsyncPeerState(push_sum_weight=42.0)
        ctx.peers["n1"] = existing_peer

        await stage._create_peer(ctx, source="n1")
        assert ctx.peers["n1"].push_sum_weight == 42.0  # not overwritten

    @pytest.mark.asyncio
    async def test_create_peer_sets_ready_when_all_nodes_started(self):
        """_create_peer sets the _nodes_ready event when all peers are present."""
        ctx = _make_ctx(neighbors=["n1"])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        # Create self peer first
        await stage._create_peer(ctx, source=ctx.address)
        assert not stage._nodes_ready.is_set()

        # Create the one neighbor
        await stage._create_peer(ctx, source="n1")
        assert stage._nodes_ready.is_set()

    @pytest.mark.asyncio
    async def test_handle_node_initialized(self):
        """handle_node_initialized delegates to _create_peer."""
        ctx = _make_ctx(neighbors=["n1"])
        stage = SetupStage()
        _attach_stage(stage, ctx)

        await stage.handle_node_initialized("n1", 0)
        assert "n1" in ctx.peers

    def test_all_nodes_started_true(self):
        """_all_nodes_started returns True when peers count == neighbors + 1."""
        ctx = _make_ctx(neighbors=["n1", "n2"])
        stage = SetupStage()
        _attach_stage(stage, ctx)
        ctx.peers = {ctx.address: AsyncPeerState(), "n1": AsyncPeerState(), "n2": AsyncPeerState()}

        assert stage._all_nodes_started(ctx) is True

    def test_all_nodes_started_false(self):
        """_all_nodes_started returns False when not all peers are present."""
        ctx = _make_ctx(neighbors=["n1", "n2"])
        stage = SetupStage()
        _attach_stage(stage, ctx)
        ctx.peers = {ctx.address: AsyncPeerState()}

        assert stage._all_nodes_started(ctx) is False

    @pytest.mark.asyncio
    async def test_run_handles_broadcast_error(self):
        """SetupStage handles broadcast errors gracefully during initialization loop."""
        ctx = _make_ctx(neighbors=[])
        stage = SetupStage()
        _attach_stage(stage, ctx)
        ctx.cp.broadcast_gossip = AsyncMock(side_effect=RuntimeError("network down"))

        with patch("p2pfl.workflow.async_dfl.stages.setup.Settings") as mock_settings:
            mock_settings.training.SYNCHRONIZATION_TIMEOUT = 0.1
            result = await stage.run()

        assert result == "training_round"
