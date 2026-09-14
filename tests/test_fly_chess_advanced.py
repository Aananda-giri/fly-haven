"""Behavioral checks for curricula, ranked sensory input and resumable experiments."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import chess
import numpy as np
import pytest
import torch

from test_fly_chess_v2 import definitions
from flychess.advanced import (AdvancedTrainer, TeacherStore, advanced_config, benchmark_advanced,
                              ingest_advanced, outcome_targets, policy_distribution,
                              prepare_advanced_data, replay_board, soft_policy_loss, stratified_pool)
from flychess.ranked import encode_ranked_board, ranked_lookahead, RANKED_FEATURES

ROOT = Path(__file__).resolve().parents[1]

def defs(*tags,name='V4_V5'):
    return definitions(*tags,notebook=ROOT/f'fly_chess_colab_{name}.ipynb')


def test_history_survives_identical_fen_and_family_split_is_stable():
    ns=defs('encoding')
    board=chess.Board()
    for move in ['g1f3','g8f6','f3g1','f6g8']*2:board.push_uci(move)
    unknown=chess.Board(board.fen(en_passant='fen'))
    assert ns['encode_board'](board).shape==(788,)
    assert not np.array_equal(ns['encode_board'](board),ns['encode_board'](unknown))
    assert ns['position_key'](board)==ns['position_key'](unknown)
    assert ns['partition'](board)==ns['partition'](unknown)
    assert ns['encode_board'](board)[782]==1 and ns['encode_board'](unknown)[783]==0


def test_soft_labels_reward_nearly_equivalent_moves_and_ignore_padding():
    logits=torch.tensor([[1.,1.,-float('inf')]],requires_grad=True)
    loss=soft_policy_loss(logits,[[0,1,2]],[[.5,.5,0.]])
    assert loss.item()==pytest.approx(np.log(2))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError,match='illegal'):soft_policy_loss(logits,[[2]],[[1.]])
    with pytest.raises(ValueError,match='sum'):soft_policy_loss(logits,[[0]],[[.1]])


@pytest.mark.parametrize('count',[1,2,32,1000])
def test_curriculum_is_unique_integer_training_only_even_small_shards(count):
    groups=[np.arange(i*100,(i+1)*100) for i in range(4)]
    a=stratified_pool(groups,count,np.random.RandomState(1),'consolidation')
    b=stratified_pool(groups,count,np.random.RandomState(1),'consolidation')
    assert a.dtype==np.int64 and len(a)==min(count,400)
    assert len(np.unique(a))==len(a) and set(a)<=set(range(400))
    np.testing.assert_array_equal(a,b)
    if count==32:
        assert any(a<100) and any((a>=100)&(a<200)) and any((a>=200)&(a<300))
        assert not np.array_equal(a,stratified_pool(groups,count,np.random.RandomState(2),'consolidation'))


def test_source_multipv_black_scores_and_labels_use_mover_perspective():
    ns=defs('encoding');board=chess.Board();board.push_uci('e2e4')
    legal=list(board.legal_moves)
    raw=json.dumps({'fen':board.fen(),'evals':[{'depth':20,'pvs':[
        {'cp':-100,'line':legal[0].uci()},{'cp':-90,'line':legal[1].uci()}]}]})
    rows,rejected=ingest_advanced([raw],1,14,ns)
    assert rejected==0 and len(rows)==1
    assert rows[0][2]>.5 and rows[0][-1][0]>rows[0][-1][1]>0 and rows[0][-1][2]==0
    assert sum(rows[0][-1])==pytest.approx(1)


def test_teacher_snapshot_survives_vm_loss_and_rejects_changed_settings(tmp_path):
    store=TeacherStore(tmp_path/'vm/teacher.sqlite',tmp_path/'drive/teacher.sqlite',{'nodes':50000})
    label={'moves':['e2e4'],'probs':[1.],'value':.5}
    store.put('fen',label);store.mark_stage('shard');store.snapshot();store.close()
    (tmp_path/'vm/teacher.sqlite').unlink()
    restored=TeacherStore(tmp_path/'vm/teacher.sqlite',tmp_path/'drive/teacher.sqlite',{'nodes':50000})
    assert restored.get('fen')==label and restored.stage_done('shard');restored.close()
    with pytest.raises(ValueError,match='configuration'):TeacherStore(tmp_path/'vm/teacher.sqlite',tmp_path/'drive/teacher.sqlite',{'nodes':1})


def test_completed_outcomes_are_mover_relative_and_replay_keeps_history():
    records=[{'turn':chess.WHITE,'start_fen':chess.STARTING_FEN,'history':'e2e4 e7e5'}, {'turn':chess.BLACK}]
    assert [r['value'] for r in outcome_targets(records,chess.WHITE)]==[1.,0.]
    assert [r['value'] for r in outcome_targets(records,None)]==[.5,.5]
    assert len(replay_board(records[0]).move_stack)==2


def test_ranker_finds_forced_mate_and_never_mutates_board():
    board=chess.Board('7k/5K2/6Q1/8/8/8/8/8 w - - 0 1');fen=board.fen()
    ranked=ranked_lookahead(fen)
    assert len(ranked)==board.legal_moves.count() and ranked[0][1]==10000
    board.push_uci(ranked[0][0]);assert board.is_checkmate()
    assert ranked==ranked_lookahead(fen)


def test_ranked_child_boards_and_candidates_flow_through_sensory_features():
    ns=defs('encoding');board=chess.Board();original=board.fen()
    features=encode_ranked_board(board,ns['encode_history_board'],ns['encode_base_board'])
    assert features.shape==(RANKED_FEATURES,) and board.fen()==original
    ranked=ranked_lookahead(original)
    move=chess.Move.from_uci(ranked[0][0]);child=board.copy();child.push(move);child.turn=board.turn
    np.testing.assert_array_equal(features[788:1568],ns['encode_base_board'](child))
    assert features[788+780+move.from_square]==1 and features[788+844+move.to_square]==1
    assert features[788+909]==1
    no=encode_ranked_board(board,ns['encode_history_board'],ns['encode_base_board'],'no_candidates')
    np.testing.assert_array_equal(no[:788],features[:788]);assert not no[788:].any()
    zero=encode_ranked_board(board,ns['encode_history_board'],ns['encode_base_board'],'zero_all')
    assert not zero.any()


def trainer_fixture(tmp_path,branches=('v4','v5'),device='cpu'):
    torch.set_num_threads(2)
    ns=defs('encoding','recovery','brain','models','control_flyonly','training_definitions','search','matches')
    ns.update(ROOT=tmp_path,RUN_DIR=tmp_path/'run',RUN_ID='test',SEED=0,USE_REAL_GRAPH=False,
              STOCKFISH_BINARY_SHA='fake',N_NEURONS=8,GRAFT_KW={'d_model':8,'n_latents':4,'n_layers':1,'n_heads':2},
              GAIN=1.,ALPHA=.5,T_P=7,T_A=3,RELAY_STEPS=3,EDGE_MIN_SYNAPSES=3,BATCH_SIZE=4,
              MODEL_SOURCE_SHA256='test-source',STOP_ROOT=tmp_path/'run',DEVICE=torch.device(device),ids=np.arange(8))
    ns['RUN_DIR'].mkdir(exist_ok=True)
    ns['sha256']=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    matrix=torch.tensor([[.1,.2,.1,.2,0,0,0,0],[.2,.1,.2,.1,0,0,0,0],
        [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[.2,.1,.2,.1,.1,0,0,0],
        [.1,.2,.1,.2,0,.1,0,0],[0,0,0,0,.3,.1,0,0],[0,0,0,0,.1,.3,0,0]]).to_sparse().coalesce().to(device)
    ns['brain']=ns['FlyBrain'](matrix,1.,.5)
    ns.update(SENSORY_IDX=np.arange(4),RELAY_IDX=np.arange(4),PREMOTOR_IDX=np.array([4,5]),MOTOR_IDX=np.array([6,7]),
              ACTIVE_MASK=torch.tensor([False]*4+[True]*4,device=device))
    projection=torch.tensor(np.random.RandomState(2).rand(8,788)*.003,dtype=torch.float32,device=device)
    ns['sensory_current']=lambda features: projection@features.t()
    def factory():
        return ns['FlyChessModel'](ns['brain'],ns['RELAY_IDX'],ns['PREMOTOR_IDX'],ns['MOTOR_IDX'],ns['ACTIVE_MASK'],7,3,**ns['GRAFT_KW']).to(device)
    ns['make_opening_pairs']=lambda count,seed=0:[['e2e4','e7e5','g1f3','b8c6','f1b5','a7a6','b5a4','g8f6']]*count
    ns['make_advanced_model']=factory
    torch.manual_seed(0);ns['model']=factory()
    rng=np.random.RandomState(22);board=chess.Board();boards=[]
    for _ in range(300):
        if board.is_game_over(claim_draw=True) or len(board.move_stack)>40:board=chess.Board()
        board.push(list(board.legal_moves)[rng.randint(board.legal_moves.count())]);boards.append(board.copy())
    fens=np.array([b.fen(en_passant='fen') for b in boards]);moves=np.array([next(iter(b.legal_moves)).uci() for b in boards])
    values=np.array([(.2,.5,.8)[i%3] for i in range(len(boards))],np.float32)
    data={'fen':fens,'move':moves,'value':values,'is_mate':np.zeros(len(boards),bool),'raw_mate':np.zeros(len(boards),int),
          'policy_moves':np.repeat(moves[:,None],3,1),'policy_probs':np.tile([1.,0.,0.],(len(boards),1))}
    path=tmp_path/'data.npz';np.savez_compressed(path,**data)
    splits=np.array([ns['partition'](b) for b in boards])
    ns.update(positions=data,positions_path=path,fens=fens,labels_uci=moves,values=values,
              train_idx=np.flatnonzero(splits=='train'),validation_idx=np.flatnonzero(splits=='validation'),test_idx=np.flatnonzero(splits=='test'))
    config=advanced_config(branches,True)
    config.update(pool_size=16,validation_size=4,pretrain_rounds=20,smoke_rounds=2)
    trainer=AdvancedTrainer(ns,config)
    def relabel(indices,deadline,limit=None):
        for i in list(indices)[:limit or config['relabel_per_round']]:
            b=chess.Board(str(fens[i]));trainer.teacher.put(b.fen(en_passant='fen'),
                {'moves':[str(moves[i])],'probs':[1.],'value':float(values[i]),'value_source':'test_wdl'})
        trainer.teacher.snapshot()
    trainer.relabel=relabel
    ns['learning_preflight']=lambda model,cache:ns['atomic_json'](ns['RUN_DIR']/'learning-preflight.json',{'passed':True})
    return ns,trainer


def test_paired_supervision_is_identical_and_resume_keeps_optimizer_rng(tmp_path):
    ns,trainer=trainer_fixture(tmp_path)
    trainer.run(.5)
    a,b=trainer.states.values()
    assert a['step']==b['step']==4 and a['round']==b['round']==2
    for key in a['model'].state_dict():torch.testing.assert_close(a['model'].state_dict()[key],b['model'].state_dict()[key],rtol=0,atol=0)
    for state in trainer.states.values():
        saved=ns['load_checkpoint'](state['directory']/'main.pt',state['identity'])
        assert saved['optimizer']['state'] and saved['scaler'] is not None
        assert saved['sampler'][2]==state['sampler'].get_state()[2]
        assert (state['directory']/saved['replay_file']).exists()
        torch.testing.assert_close(saved['rng']['torch'],state['rng']['torch'])
    trainer.teacher.close()
    _,resumed=trainer_fixture(tmp_path)
    resumed.run(.5)
    assert {s['step'] for s in resumed.states.values()}=={4}
    resumed.teacher.close()


def test_capped_selfplay_never_enters_replay_as_draw(tmp_path):
    ns,trainer=trainer_fixture(tmp_path,('v5',))
    state=trainer.states['v5'];trainer.config.update(game_cap=9,games_per_round=1)
    assert trainer.collect_selfplay(state,time.monotonic()+20)
    assert state['selfplay_games']==1 and not state['replay']
    game=json.loads((state['directory']/'selfplay-games/000001.json').read_text())
    assert game['status']=='unresolved' and game['result'] is None and game['train_records']==0
    assert not (state['directory']/'selfplay-in-progress.json').exists()
    trainer.teacher.close()


def test_notebooks_compile_and_pinned_implementation_is_verifiable():
    import hashlib
    for name in ['V4','V5','V4_V5','V5_5']:
        doc=json.loads((ROOT/f'fly_chess_colab_{name}.ipynb').read_text())
        for cell in doc['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']));assert cell['outputs']==[]
        controls=next(c for c in doc['cells'] if c['metadata'].get('tags')==['controls'])
        assigns={n.targets[0].id:n.value for n in ast.parse(''.join(controls['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        pinned=json.loads(ast.literal_eval(assigns['NOTEBOOK_SOURCE'].args[0]))
        assert hashlib.sha256(json.dumps(pinned,sort_keys=True).encode()).hexdigest()==ast.literal_eval(assigns['MODEL_SOURCE_SHA256'])
        advanced=next(c for c in doc['cells'] if c['metadata'].get('tags')==['advanced_definitions'])
        assert ''.join(advanced['source']).strip()==(ROOT/'flychess/advanced.py').read_text().strip()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_amp_preserves_sparse_physics_and_finite_graft_gradients(tmp_path):
    ns,trainer=trainer_fixture(tmp_path,('v4',),device='cuda')
    state=trainer.states['v4'];records=[trainer.source_record(i) for i in ns['train_idx'][:4]]
    features=torch.tensor(np.stack([ns['encode_board'](replay_board(r)) for r in records]),device='cuda')
    full=state['model'].perceive(features)
    with torch.autocast('cuda',dtype=torch.float16):
        half_context=state['model'].perceive(features)
    for a,b in zip(full,half_context):torch.testing.assert_close(a,b,rtol=0,atol=0);assert b.dtype==torch.float32
    loss=trainer.loss_batch(state,records)
    assert np.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in state['model'].graft.parameters())
    trainer.teacher.close()


def test_finished_selfplay_uses_real_outcome_and_resumes_between_plies(tmp_path,monkeypatch):
    import flychess.advanced as module
    ns,trainer=trainer_fixture(tmp_path,('v5',));state=trainer.states['v5']
    trainer.config.update(game_cap=50,games_per_round=1)
    ns['partition']=lambda board:'train'
    ns['make_opening_pairs']=lambda count,seed=0:[['f2f3','e7e5']]
    def forced(ns,model,board,simulations,rng,exploration=False):
        move=chess.Move.from_uci('g2g4' if board.turn else 'd8h4')
        return [ns['move_index'](move,board.turn)],np.array([1.],np.float32),None
    monkeypatch.setattr(module,'search_target',forced)
    calls=[0]
    def interrupt():
        calls[0]+=1
        if calls[0]==2:raise ns['RunPaused']('simulated reset between plies')
    trainer.save(state);ns['check_stop']=interrupt
    with pytest.raises(ns['RunPaused']):trainer.collect_selfplay(state,time.monotonic()+10)
    assert (state['directory']/'selfplay-in-progress.json').exists()
    trainer.teacher.close()
    ns,resumed=trainer_fixture(tmp_path,('v5',));state=resumed.states['v5']
    resumed.config.update(game_cap=50,games_per_round=1);ns['partition']=lambda board:'train'
    assert resumed.collect_selfplay(state,time.monotonic()+10)
    assert state['selfplay_games']==1 and len(state['replay'])==2
    assert [r['value'] for r in state['replay']]==[0.,1.]
    assert [len(replay_board(r).move_stack) for r in state['replay']]==[2,3]
    assert not (state['directory']/'selfplay-in-progress.json').exists()
    resumed.teacher.close()


def test_selfplay_excludes_heldout_position_families(tmp_path,monkeypatch):
    import flychess.advanced as module
    ns,trainer=trainer_fixture(tmp_path,('v5',));state=trainer.states['v5']
    ns['partition']=lambda board:'test'
    ns['make_opening_pairs']=lambda count,seed=0:[['f2f3','e7e5','g2g4']]
    monkeypatch.setattr(module,'search_target',lambda ns,m,b,s,r,e=False:([ns['move_index'](chess.Move.from_uci('d8h4'),b.turn)],np.array([1.],np.float32),None))
    assert trainer.collect_selfplay(state,time.monotonic()+10) and not state['replay']
    game=json.loads((state['directory']/'selfplay-games/000001.json').read_text())
    assert game['status']=='complete' and game['result']=='0-1' and game['train_records']==0
    trainer.teacher.close()


def test_benchmark_interleaves_branches_pins_snapshots_and_resumes(tmp_path,monkeypatch):
    import contextlib
    import flychess.advanced as module
    ns,trainer=trainer_fixture(tmp_path)
    trainer.run(.5)
    monkeypatch.setattr(module,'clone_checkpoint_model',lambda ns,state,path:state['model'])
    class Engine:
        options={'UCI_Elo':SimpleNamespace(min=1320,max=3190)}
        def configure(self,config):assert config['UCI_LimitStrength'] and config['Threads']==1
    @contextlib.contextmanager
    def engine():yield Engine()
    ns['open_stockfish']=engine
    calls=[]
    def game(white,black,opening,game_id,**kwargs):
        calls.append((str(ns['RUN_DIR']),game_id));assert kwargs['initial_clock']==120 and kwargs['increment']==1
        return {'score':None,'reason':'safety_cap'}
    ns['play_game']=game
    benchmark_advanced(trainer,.5,1)
    assert len(calls)==6
    assert '/v4/' in calls[0][0] and '/v5/' in calls[2][0] and 'network_raw' in calls[4][0]
    reports=list(tmp_path.rglob('strength/**/summary.json'));assert len(reports)==3
    for path in reports:
        report=json.loads(path.read_text());assert report['reference_scale_estimate'] is None and report['unresolved']==2
        assert report['configuration']['assist_mode']=='none'
    configs=[json.loads(p.read_text()) for p in tmp_path.rglob('strength/**/config.json')]
    assert sorted(c['simulations'] for c in configs)==[0,0,4]
    benchmark_advanced(trainer,.5,1);assert len(calls)==6
    trainer.teacher.close()


def test_partial_round_resume_matches_uninterrupted_supervision(tmp_path):
    (tmp_path/'resume').mkdir()
    ns,trainer=trainer_fixture(tmp_path/'resume')
    def interrupt():
        if trainer.states['v4']['step']==1:raise ns['RunPaused']('test interruption')
    ns['check_stop']=interrupt
    trainer.run(.5)
    assert trainer.states['v4']['step']==1 and trainer.states['v4']['round']==0
    trainer.teacher.close()
    ns,resumed=trainer_fixture(tmp_path/'resume');resumed.run(.5)
    (tmp_path/'reference').mkdir();_,reference=trainer_fixture(tmp_path/'reference');reference.run(.5)
    for branch in ('v4','v5'):
        for key,value in resumed.states[branch]['model'].state_dict().items():
            torch.testing.assert_close(value,reference.states[branch]['model'].state_dict()[key],rtol=0,atol=0)
    resumed.teacher.close();reference.teacher.close()


def test_existing_multipv_dataset_is_copied_without_flattening(tmp_path,monkeypatch):
    ns,trainer=trainer_fixture(tmp_path)
    data=copy.deepcopy(ns['positions']);data['split']=np.array([ns['partition'](chess.Board(str(f))) for f in data['fen']])
    data['policy_probs'][:,0]=.6;data['policy_probs'][:,1]=.4
    source=tmp_path/'multipv.npz';np.savez_compressed(source,**data)
    ns['atomic_json'](source.with_suffix('.json'),{'schema':4,'sha256':ns['sha256'](source)})
    ns['CHESS_DATA']=tmp_path/'advanced-data';ns['CHESS_DATA'].mkdir()
    monkeypatch.setenv('FLY_CHESS_SOURCE_DATASET',str(source));monkeypatch.setenv('FLY_CHESS_POSITIONS',str(len(data['fen'])))
    new=prepare_advanced_data(ns)
    np.testing.assert_array_equal(new['policy_probs'],data['policy_probs'])
    np.testing.assert_array_equal(new['split'],data['split'])
    pin=json.loads(ns['positions_path'].with_suffix('.json').read_text())
    assert pin['origin']['policy']=='copied_source_multipv'
    trainer.teacher.close()


def test_failed_preflight_cannot_be_bypassed_by_resuming(tmp_path):
    ns,trainer=trainer_fixture(tmp_path,('v4',));trainer.smoke=False
    state=trainer.states['v4'];ns['atomic_json'](state['directory']/'learning-preflight.json',{'passed':False})
    with pytest.raises(RuntimeError,match='remains stopped'):trainer.run(.5)
    assert state['step']==0
    trainer.teacher.close()
