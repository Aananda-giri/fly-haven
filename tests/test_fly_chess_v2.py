"""Behavioral tests execute notebook definitions without data or training side effects."""
import ast
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import chess
import chess.engine
import chess.pgn
import pickle
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

NOTEBOOK = Path(__file__).resolve().parents[1] / 'fly-chess.ipynb'


def definitions(*tags):
    namespace = dict(chess=chess, np=np, torch=torch, nn=nn, F=F, math=math,
                     contextlib=contextlib, hashlib=hashlib, random=random, os=os,
                     Path=Path, json=json, time=time, pickle=pickle, DEVICE=torch.device('cpu'),
                     check_stop=lambda: None, MODE='smoke')
    notebook = json.loads(NOTEBOOK.read_text())
    for tag in tags:
        cell = next(c for c in notebook['cells'] if tag in c['metadata'].get('tags', []))
        module = ast.parse(''.join(cell['source']))
        if tag in ('encoding', 'search'):
            nodes = module.body
        else:
            nodes = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return namespace


def test_exact_promotions_and_both_color_roundtrip():
    ns = definitions('encoding')
    for fen in ('7k/P7/8/8/8/8/8/7K w - - 0 1',
                '7k/8/8/8/8/8/p7/7K b - - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '7k/8/8/3pP3/8/8/8/7K w - d6 0 1'):
        board = chess.Board(fen)
        indices = [ns['move_index'](m, board.turn) for m in board.legal_moves]
        assert len(indices) == len(set(indices))
        for move in board.legal_moves:
            assert ns['index_to_move'](ns['move_index'](move, board.turn), board) == move
    assert ns['N_MOVES'] == 4184


def test_label_perspective_and_terminal_values():
    ns = definitions('encoding')
    board = chess.Board()
    assert ns['source_value'](board, cp=500) > .8
    board.turn = chess.BLACK
    assert ns['source_value'](board, cp=500) < .2
    assert ns['source_value'](board, mate=3) == 0
    assert ns['source_value'](board, mate=-3) == 1
    mated = chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
    assert ns['terminal_value'](mated) == 0
    assert ns['source_value'](mated, mate=0) == 0
    assert ns['terminal_value'](chess.Board('7k/5K2/6Q1/8/8/8/8/8 b - - 0 1')) == .5


def test_partition_groups_identical_inputs():
    ns = definitions('encoding')
    a, b = chess.Board(), chess.Board()
    b.fullmove_number = 21
    b.halfmove_clock = 19
    assert ns['position_key'](a) == ns['position_key'](b)
    assert ns['partition'](a) == ns['partition'](b)


def test_search_opponent_perspective_and_backup():
    ns = definitions('encoding', 'training_definitions', 'search')
    root = ns['Node'](chess.Board())
    good, bad = ns['Node'](chess.Board(), root, .5), ns['Node'](chess.Board(), root, .5)
    good.visits = bad.visits = 10
    good.value_sum, bad.value_sum = 1, 9
    assert ns['puct_score'](good, 20) > ns['puct_score'](bad, 20)
    ns['backup']([root, good], 0)
    assert root.q() == 1
    assert good.value_sum == 1


def test_search_distinct_leaves_and_forced_mate():
    ns = definitions('encoding', 'training_definitions', 'search')
    submissions = []
    def evaluator(model, boards):
        assert len({b.fen() for b in boards}) == len(boards)
        submissions.append(len(boards))
        priors = []
        for board in boards:
            legal = list(ns['legal_move_table'](board))
            priors.append({idx: 1 / len(legal) for idx in legal})
        return priors, [.5] * len(boards)
    ns['evaluate_leaves'] = evaluator
    root = ns['search'](None, chess.Board(), 32, batch=8)
    assert root.visits == 32
    assert max(submissions) > 1
    assert root.pending == 0
    board = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    move, tree = ns['best_move_by_search'](None, board, 256)
    board.push(move)
    assert board.is_checkmate()
    assert tree.visits == 256


def test_terminal_search_never_calls_network_and_preserves_repetition():
    ns = definitions('encoding', 'training_definitions', 'search')
    ns['evaluate_leaves'] = lambda *a: pytest.fail('terminal network call')
    board = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
        board.push_uci(move)
    assert ns['terminal_value'](board) == .5
    tree = ns['search'](None, board)
    assert tree.board.move_stack == board.move_stack
    assert tree.neural_evaluations == 0


def test_search_failure_releases_pending():
    ns = definitions('encoding', 'training_definitions', 'search')
    captured = []
    original_reserve = ns['reserve_leaf']
    def reserve(root):
        captured.append(root)
        return original_reserve(root)
    ns['reserve_leaf'] = reserve
    calls = 0
    def evaluate(model, boards):
        nonlocal calls
        calls += 1
        if calls > 1: raise RuntimeError('network failed')
        legal = list(ns['legal_move_table'](boards[0]))
        return [{i: 1 / len(legal) for i in legal}], [.5]
    ns['evaluate_leaves'] = evaluate
    with pytest.raises(RuntimeError): ns['search'](None, chess.Board(), 8)
    assert captured[0].pending == 0
    assert all(c.pending == 0 for c in captured[0].children.values())


def test_active_subgraph_outputs_and_gradients():
    ns = definitions('brain')
    indices = torch.tensor([[0, 1, 2, 3, 1], [1, 2, 1, 0, 3]])
    matrix = torch.sparse_coo_tensor(indices, torch.tensor([.2, -.1, .3, .2, .1]), (4, 4)).coalesce()
    brain = ns['FlyBrain'](matrix, 1.5, .5)
    initial = torch.full((4, 2), .3)
    active = np.array([1, 2])
    current = torch.full((2, 2), .1, requires_grad=True)
    full_current = torch.zeros(4, 2)
    full_current[active] = current
    mask = torch.tensor([False, True, True, False])
    reference = brain.run_clamped(initial, full_current, mask, 3)[active]
    optimized = brain.run_active(initial, current, active, brain.active_blocks(active), 3)
    torch.testing.assert_close(reference, optimized)
    reference_grad = torch.autograd.grad(reference.sum(), current, retain_graph=True)[0]
    optimized_grad = torch.autograd.grad(optimized.sum(), current)[0]
    torch.testing.assert_close(reference_grad, optimized_grad)


def test_linear_decoder_and_current_gradient():
    ns = definitions('encoding', 'models')
    decoder = ns['MoveDecoder'](8)
    assert all(isinstance(m, nn.Linear) for m in decoder.children())
    assert not any(isinstance(m, nn.GELU) for m in decoder.modules())


def test_atomic_checkpoint_rng_resume_and_mismatch(tmp_path):
    ns = definitions('recovery')
    ns['DEVICE'] = torch.device('cpu')
    path = tmp_path / 'checkpoint.pt'
    sampler = np.random.RandomState(4)
    before = ns['rng_state'](sampler)
    expected = sampler.randint(1000, size=20)
    ns['save_checkpoint'](path, {'identity': 'one', 'rng': before, 'step': 1})
    state = ns['load_checkpoint'](path, 'one')
    ns['restore_rng'](state['rng'], sampler)
    np.testing.assert_array_equal(sampler.randint(1000, size=20), expected)
    ns['save_checkpoint'](path, {'identity': 'one', 'rng': before, 'step': 2})
    path.write_bytes(b'broken')
    assert ns['load_checkpoint'](path, 'one')['step'] == 1
    with pytest.raises(ValueError): ns['load_checkpoint'](path, 'two')


def test_root_score_mover_perspective():
    ns = definitions('encoding', 'scoring')
    ns.update(SCORE_CACHE={}, STOCKFISH_DEPTH=3, STOCKFISH_BINARY_SHA='test',
              SCORE_CACHE_PATH=Path('/unused'), atomic_json=lambda *a: None)
    class Engine:
        def configure(self, options): pass
        def analyse(self, board, limit, root_moves):
            return {'score': chess.engine.PovScore(chess.engine.Cp(300), board.turn)}
    for turn in (chess.WHITE, chess.BLACK):
        board = chess.Board(); board.turn = turn
        original = board.fen()
        score = ns['score_move'](Engine(), board, next(iter(board.legal_moves)))
        assert score['cp'] == 300
        assert board.fen() == original
    scores = {'e2e4': {'cp': 300, 'mate': None, 'expectation': .8},
              'e2e3': {'cp': -100, 'mate': None, 'expectation': .3}}
    assert ns['position_loss'](scores, chess.Move.from_uci('e2e4'))['cp_loss'] == 0
    assert ns['position_loss'](scores, chess.Move.from_uci('e2e3'))['cp_loss'] == 400


def test_safety_cap_unresolved_and_timeout(tmp_path):
    ns = definitions('encoding', 'matches')
    ns.update(RUN_DIR=tmp_path, atomic_json=lambda path, value: Path(path).write_text(json.dumps(value)))
    mover = lambda board, clock, inc: next(iter(board.legal_moves))
    game = ns['play_game'](mover, mover, [], 'cap', max_plies=2)
    assert game['score'] is None
    assert game['reason'] == 'safety_cap'
    def slow(board, clock, inc):
        time.sleep(.01)
        return next(iter(board.legal_moves))
    game = ns['play_game'](slow, mover, [], 'timeout', initial_clock=.001)
    assert game['score'] == 0
    assert game['reason'] == 'timeout'


def test_ingestion_normalizes_castling_and_rejects_illegal_labels():
    ns = definitions('encoding', 'data')
    white = {'fen': 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
             'evals': [{'depth': 20, 'pvs': [{'cp': 200, 'line': 'e1h1'}]}]}
    black = {'fen': 'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
             'evals': [{'depth': 20, 'pvs': [{'cp': 200, 'line': 'e8h8'}]}]}
    illegal = {'fen': chess.STARTING_FEN,
               'evals': [{'depth': 20, 'pvs': [{'cp': 200, 'line': 'e2e7'}]}]}
    rows, rejected = ns['ingest_positions']([json.dumps(x) for x in (white, black, illegal)], 10, 10)
    assert [row[1] for row in rows] == ['e1g1', 'e8g8']
    assert rows[0][2] > .5 > rows[1][2]
    assert sum(rejected.values()) == 1


def test_shuffle_preserves_degrees_sign_classes_and_selfloops():
    from scipy.sparse import csr_matrix
    ns = definitions('controls')
    ns['EDGE_MIN_SYNAPSES'] = 3
    generator = np.random.RandomState(1)
    a = generator.randint(0, 20, 120)
    b = generator.randint(0, 20, 120)
    graph = csr_matrix((np.full(120, 4, np.float32), (a, b)), shape=(20, 20))
    signs = np.where(np.arange(20) % 2, 1, -1)
    result, swaps = ns['shuffled_graph'](graph, signs)
    assert swaps > 0
    np.testing.assert_array_equal(np.diff(graph.indptr), np.diff(result.indptr))
    np.testing.assert_array_equal(np.diff(graph.tocsc().indptr), np.diff(result.tocsc().indptr))
    np.testing.assert_array_equal(graph.diagonal(), result.diagonal())
    for sign in (-1, 1):
        np.testing.assert_array_equal(np.diff(graph[signs == sign].tocsc().indptr),
                                      np.diff(result[signs == sign].tocsc().indptr))
    again, _ = ns['shuffled_graph'](graph, signs)
    assert (again != result).nnz == 0


def test_resume_matches_uninterrupted_training(tmp_path):
    ns = definitions('encoding', 'recovery', 'training_definitions')
    ns.update(BATCH_SIZE=2, train_idx=np.arange(4),
              fens=np.array([chess.STARTING_FEN] * 4),
              labels_uci=np.array(['e2e4', 'd2d4', 'g1f3', 'b1c3']),
              values=np.array([.5, .6, .4, .5], np.float32))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(780, 4185)
        def forward(self, features):
            output = self.layer(features)
            return output[:, :-1], output[:, -1]
    torch.manual_seed(7)
    continuous = Model()
    interrupted = Model(); interrupted.load_state_dict(continuous.state_dict())
    def initialize(model):
        return torch.optim.AdamW(model.parameters(), lr=3e-4), np.random.RandomState(0)
    optimizer, sampler = initialize(continuous)
    for _ in range(4): ns['training_update'](continuous, optimizer, sampler)
    optimizer, sampler = initialize(interrupted)
    for _ in range(2): ns['training_update'](interrupted, optimizer, sampler)
    path = tmp_path / 'training.pt'
    ns['save_checkpoint'](path, {'identity': 'test', 'model': interrupted.state_dict(),
              'optimizer': optimizer.state_dict(), 'rng': ns['rng_state'](sampler), 'step': 2})
    restarted = Model(); optimizer, sampler = initialize(restarted)
    state = ns['load_checkpoint'](path, 'test')
    restarted.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    ns['restore_rng'](state['rng'], sampler)
    for _ in range(2): ns['training_update'](restarted, optimizer, sampler)
    for actual, expected in zip(restarted.parameters(), continuous.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_interventions_preserve_or_remove_expected_paths():
    ns = definitions('encoding', 'brain', 'models')
    ns['N_NEURONS'] = 8
    injection = torch.zeros(8, 780); injection[0, 0] = 1
    ns['sensory_current'] = lambda features: injection @ features.t()
    indices = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6]])
    matrix = torch.sparse_coo_tensor(indices, torch.ones(7), (8, 8)).coalesce()
    brain = ns['FlyBrain'](matrix, 1, .5)
    model = ns['FlyChessModel'](brain, np.array([2, 3]), np.array([4, 5]), np.array([6, 7]),
        torch.tensor([False] * 4 + [True] * 4), 8, 3,
        d_model=8, n_latents=4, n_layers=1, n_heads=2)
    features = torch.zeros(2, 780); features[0, 0] = 1
    zero_senses, _ = model(features, intervention='no_senses')
    torch.testing.assert_close(zero_senses[0], zero_senses[1])
    no_graft, _ = model(features, intervention='no_graft')
    assert not torch.equal(no_graft[0], no_graft[1])
    intact, _ = model(features)
    intact.sum().backward()
    assert any(p.grad is not None for p in model.graft.parameters())
    assert list(model.brain.parameters()) == []


def test_assists_are_separate_and_logged():
    ns = definitions('encoding', 'training_definitions', 'search')
    board = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    bad = chess.Move.from_uci('g6g5')
    logits = torch.zeros(1, 4184); logits[0, ns['move_index'](bad, board.turn)] = 10
    ns['predict'] = lambda model, boards: (logits, torch.tensor([.5]))
    plain, metadata = ns['choose_move'](None, board, return_metadata=True)
    assert plain == bad and not metadata['assisted']
    assisted, metadata = ns['choose_move'](None, board, assist_mode='mate1', return_metadata=True)
    board.push(assisted)
    assert board.is_checkmate() and metadata['assisted']
    assert metadata['model_move'] == bad.uci()


def test_puzzle_overlap_checks_decision_not_setup():
    ns = definitions('encoding', 'puzzles_data')
    row = {'FEN': chess.STARTING_FEN, 'Moves': 'e2e4 e7e5 g1f3 b8c6'}
    board = chess.Board(); board.push_uci('e2e4')
    assert ns['puzzle_training_overlap'](row, {ns['position_key'](board)})
    assert not ns['puzzle_training_overlap'](row, {ns['position_key'](chess.Board())})


def test_pair_summary_excludes_unresolved_and_partial_pairs():
    ns = definitions('scoring', 'matches')
    summary = ns['pair_score_summary']([
        {'pair': 0, 'score': 1}, {'pair': 0, 'score': .5},
        {'pair': 1, 'score': None}, {'pair': 1, 'score': 0}])
    assert summary['paired_score']['mean'] == .75
    assert summary['paired_score']['n'] == 1
    assert summary['unresolved'] == 1


def test_ratio_undefined_without_random_loss_and_matched_denominator():
    ns = definitions('scoring')
    assert ns['ratio_summary']([0, 0], [0, 0])['value'] is None
    assert ns['ratio_summary']([.1, .1], [.2, .2])['value'] == pytest.approx(.5)
