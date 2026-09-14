"""Behavioral regression tests for the exact standalone V3 notebook definitions."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import contextlib

import chess
import numpy as np
import pytest
import torch

from test_fly_chess_v2 import definitions
from flychess.strength import strength_summary

NOTEBOOK = Path(__file__).resolve().parents[1] / "fly_chess_colab_V3.ipynb"


def v3(*tags):
    return definitions(*tags, notebook=NOTEBOOK)


def tiny_model():
    ns = v3("encoding", "brain", "models", "training_definitions")
    ns["N_NEURONS"] = 8
    injection = torch.zeros(8, 780)
    injection[:4, :4] = torch.eye(4)
    ns["sensory_current"] = lambda features: injection @ features.t()
    matrix = torch.tensor([[.1, .2, .1, .2, 0, 0, 0, 0],
                           [.2, .1, .2, .1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [.2, .1, .2, .1, .1, 0, 0, 0],
                           [.1, .2, .1, .2, 0, .1, 0, 0],
                           [0, 0, 0, 0, .3, .1, 0, 0],
                           [0, 0, 0, 0, .1, .3, 0, 0]]).to_sparse().coalesce()
    brain = ns["FlyBrain"](matrix, 1, .5)
    model = ns["FlyChessModel"](brain, np.array([0, 1, 2, 3]), np.array([4, 5]),
        np.array([6, 7]), torch.tensor([False]*4 + [True]*4), 7, 3,
        d_model=8, n_latents=4, n_layers=1, n_heads=2)
    model.eval()
    return ns, model


def test_all_notebook_cells_compile_and_outputs_cleared():
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []


def test_prepared_frozen_states_preserve_outputs_and_graft_gradients():
    ns, model = tiny_model()
    features = torch.zeros(3, 780)
    features[:3, :3] = torch.eye(3)
    relay, initial, background = model.perceive(features)
    actual, value = model.forward_perceived(relay, initial, background)
    with torch.no_grad():
        perceived = model.brain.run(torch.zeros(8, 3), ns["sensory_current"](features), model.T_p)
        early = model.brain.run(torch.zeros(8, 3), ns["sensory_current"](features), model.relay_steps)
    current = torch.zeros(8, 3)
    current[model.premotor_idx] = (model.graft(early[model.relay_idx].t()) * model.current_amplitude).t()
    full = model.brain.run_clamped(perceived, current, model.active_mask, model.T_a)
    expected, expected_value = model.decoder(full[model.motor_idx].t())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(value, expected_value)
    parameters = tuple(model.graft.parameters())
    actual_grads = torch.autograd.grad(actual.sum()+value.sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum()+expected_value.sum(), parameters)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
    assert sum(g.abs().sum().item() for g in actual_grads) > 0


def test_singleton_relay_permutation_changes_signal_and_current_is_bounded():
    _, model = tiny_model()
    features = torch.zeros(1, 780); features[0, 0] = 1
    intact, _ = model(features)
    permuted, _ = model(features, intervention="relay_permute")
    assert not torch.equal(intact, permuted)
    current = model.graft(torch.randn(12, 4)*100) * model.current_amplitude
    assert current.abs().max() <= model.current_amplitude


def test_early_relay_keeps_its_snapshot_while_motor_perception_continues():
    ns, model = tiny_model()
    features = torch.zeros(2,780); features[0,0]=1; features[1,1]=1
    with torch.no_grad():
        early = model.brain.run(torch.zeros(8,2),ns["sensory_current"](features),3)
        full = model.brain.run(torch.zeros(8,2),ns["sensory_current"](features),7)
        relay, initial, background = model.perceive(features)
    torch.testing.assert_close(relay,early[model.relay_idx].t())
    torch.testing.assert_close(initial,full[model.active_idx].t())
    assert not torch.equal(relay,full[model.relay_idx].t())


def test_sensory_pool_reads_neuronal_groups_in_the_recorded_order():
    ns = v3("encoding", "models")
    mapping = torch.sparse_coo_tensor(torch.tensor([list(range(8)),[0,0,1,1,2,2,3,3]]),
                                     torch.ones(8),(8,4)).coalesce()
    order = np.array([2,0,7,4,1,3,5,6])
    pool = ns["make_relay_pool"](order,mapping)
    rates = torch.tensor([[.2,.4,.5,.7,.1,.3,.6,.8]])
    pooled = torch.sparse.mm(pool,rates[:,order].t()).t()
    torch.testing.assert_close(pooled,torch.tensor([[.3,.6,.2,.7]]))
    assert ns["make_relay_pool"](np.array([0,1]),mapping) is None
    graft = ns["CortexGraft"](order,np.array([0,1]),d_model=8,n_latents=4,
                              n_layers=1,n_heads=2,relay_pool=pool)
    assert graft.relay_summary[0].in_features == 4
    torch.testing.assert_close(graft.pool_relay(rates[:,order]),pooled)


def test_square_tokens_share_piece_channels_across_locations():
    ns = v3("encoding", "models")
    pool = torch.eye(780).to_sparse().coalesce()
    graft = ns["CortexGraft"](np.arange(780),np.array([0,1]),d_model=8,n_latents=4,
                              n_layers=1,n_heads=2,relay_pool=pool).eval()
    assert graft.square_tokens
    observed = []
    hook = graft.piece_embed.register_forward_pre_hook(lambda module,args: observed.append(args[0]))
    rates = torch.zeros(2,780)
    rates[0,64+10] = 1; rates[1,64+20] = 1
    currents = graft(rates)
    hook.remove()
    assert observed[0].shape == (2,64,12)
    torch.testing.assert_close(observed[0][0,10],observed[0][1,20])
    assert observed[0][0,10,1] == 1
    assert currents.shape == (2,2)
    assert not torch.equal(currents[0],currents[1])


def records(scores):
    return [{"pair": i//2, "color": "white" if i%2 == 0 else "black", "score": s}
            for i, s in enumerate(scores)]


def test_all_losses_give_non_degenerate_upper_bound_and_no_finite_rating():
    result = strength_summary(records([0]*40), 1320, "Stockfish UCI_Elo benchmark")
    assert result["status"] == "upper_bound"
    assert result["elo_difference"] is None
    assert result["reference_scale_estimate"] is None
    assert 0 < result["score_interval"][1] < .5
    assert result["reference_scale_interval"][0] is None
    assert result["reference_scale_interval"][1] < 1320


def test_partial_pairs_unresolved_and_duplicate_colors_excluded():
    rows = records([1, .5, None, 0, 1])
    rows += [{"pair": 9, "color": "white", "score": 1}]*2
    result = strength_summary(rows)
    assert result["complete_pairs"] == 1
    assert result["score"] == .75
    assert result["unresolved"] == 1
    assert result["reference_scale_estimate"] is None
    assert result["elo_difference"] == pytest.approx(190.8485, rel=1e-5)


def test_rating_requires_reference_source_and_module_matches_notebook():
    with pytest.raises(ValueError):
        strength_summary(records([.5,.5]), 1300)
    ns = v3("matches")
    rows = records([0]*40)
    assert ns["strength_summary"](rows, 1320, "benchmark") == strength_summary(rows, 1320, "benchmark")


def test_affine_motor_normalization_stays_linear():
    ns = v3("encoding", "models")
    decoder = ns["MoveDecoder"](4)
    decoder.eval()
    decoder.motor_mean.copy_(torch.tensor([.1,.2,.3,.4]))
    decoder.motor_scale.copy_(torch.tensor([.03,.2,.1,.3]))
    x, y = torch.randn(3,4), torch.randn(3,4)
    zero, _ = decoder(torch.zeros_like(x))
    a, _ = decoder(x); b, _ = decoder(y); together, _ = decoder(x+y)
    torch.testing.assert_close(together, a+b-zero, atol=1e-5, rtol=1e-5)


def test_full_precision_cache_can_resume_and_matches_live_inference(tmp_path):
    ns, model = tiny_model()
    features = torch.zeros(5,780)
    features[:4,:4] = torch.eye(4)
    ns.update(ROOT=tmp_path, RUN_ID="one", BATCH_SIZE=2,
              atomic_json=lambda p,v: Path(p).write_text(json.dumps(v)))
    ns["make_batch"] = lambda rows: (None, features[rows], None, None)
    calls = 0
    def interrupted():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
    ns["check_stop"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        ns["cache_perception"](model,"main",np.arange(5),"same")
    progress = tmp_path / "perception-cache/one/main/progress.json"
    assert json.loads(progress.read_text())["completed"] == 2
    ns["check_stop"] = lambda: None
    cache = ns["cache_perception"](model,"main",np.arange(5),"same")
    actual, value = ns["cached_forward"](model,cache,np.array([4,1,3]))
    expected, expected_value = model(features[[4,1,3]])
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(value,expected_value)
    with pytest.raises(ValueError):
        ns["cache_perception"](model,"main",np.arange(5),"different")


def test_c0_can_reuse_main_perception_without_another_graph_run():
    ns, model = tiny_model()
    c0_definitions = v3("encoding", "models", "control_flyonly")
    c0 = c0_definitions["FlyOnlyModel"](model.brain,np.array([6,7]),model.T_p)
    c0.eval()
    c0_definitions["sensory_current"] = ns["sensory_current"]
    features = torch.zeros(3,780); features[:3,:3]=torch.eye(3)
    relay, initial, background = model.perceive(features)
    cache = {"lookup": {0:0,1:1,2:2}, "arrays": [initial.detach().numpy()],
             "select_columns": [model.motor_active_idx]}
    actual, value = ns["cached_forward"](c0,cache,np.array([0,2]))
    expected, expected_value = c0(features[[0,2]])
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(value,expected_value)


def test_mate_on_last_allowed_ply_is_resolved(tmp_path):
    ns = v3("encoding", "matches")
    ns.update(RUN_DIR=tmp_path, atomic_json=lambda p,v: Path(p).write_text(json.dumps(v)))
    opening = ["f2f3", "e7e5", "g2g4"]
    white = lambda *args: pytest.fail("White should not move")
    black = lambda *args: chess.Move.from_uci("d8h4")
    result = ns["play_game"](white,black,opening,"last-mate",max_plies=1)
    assert result["score"] == 0
    assert result["reason"] == "CHECKMATE"


def test_exhausted_strength_stage_gets_new_allowance_on_resume(tmp_path, monkeypatch):
    ns = v3("encoding", "matches")
    class Engine:
        options = {"UCI_Elo": SimpleNamespace(min=1320,max=3190)}
        def configure(self, options): pass
    games=[]
    def game(white,black,opening,game_id,**kwargs):
        games.append(game_id)
        return {"score": .5, "reason": "THREEFOLD_REPETITION"}
    ns.update(MODE="full", RUN_DIR=tmp_path, model=None, model_c0=None,
              RunPaused=RuntimeError, STOCKFISH_BINARY_SHA="test", os=__import__("os"),
              open_stockfish=lambda: contextlib.nullcontext(Engine()), play_game=game,
              model_opponent=lambda *a: None, make_opening_pairs=lambda n: [["e2e4"]]*n,
              atomic_json=lambda p,v: Path(p).write_text(json.dumps(v)))
    monkeypatch.setenv("FLY_CHESS_MATCH_PAIRS","1")
    monkeypatch.setenv("FLY_CHESS_EVAL_MINUTES","0")
    ns["IDENTITY"]="test"
    assert ns["evaluate_strength"]()["status"] == "paused"
    assert not games
    monkeypatch.setenv("FLY_CHESS_EVAL_MINUTES","1")
    assert ns["evaluate_strength"]()["status"] == "complete"
    assert len(games)==16
    ns["evaluate_strength"]()
    assert len(games)==16


def test_early_stop_keeps_best_weights_and_closed_training_does_not_repeat(tmp_path, monkeypatch):
    ns = v3("recovery", "training")
    main, c0 = torch.nn.Linear(1,1), torch.nn.Linear(1,1)
    initial = main.weight.detach().clone()
    calls = {id(main):0,id(c0):0}
    def update(model,optimizer,sampler,cache,**kwargs):
        calls[id(model)] += 1
        with torch.no_grad(): model.weight.add_(1)
        return {"loss":1}
    def validation(model,*args,**kwargs):
        return {"validation_loss": calls[id(model)]/100}
    ns.update(models={"main":main,"c0":c0},RUN_DIR=tmp_path,IDENTITY="same",SEED=0,
              STEPS_TARGET=1000,VALUE_WEIGHT=4,VALIDATION_SAMPLE=np.arange(2),perception_caches={},
              training_update=update,move_match=validation,event=lambda value:None,
              fcntl=__import__("fcntl"))
    monkeypatch.setenv("FLY_CHESS_PATIENCE","2")
    ns["train_all"]()
    status = json.loads((tmp_path/"training-status.json").read_text())
    assert status["early_stopped"] and status["common_step"] == 300
    torch.testing.assert_close(main.weight,initial+100)
    assert ns["load_checkpoint"](tmp_path/"main.pt","same")["step"] == 300
    ns["train_all"]()
    assert calls == {id(main):300,id(c0):300}
    torch.testing.assert_close(main.weight,initial+100)
