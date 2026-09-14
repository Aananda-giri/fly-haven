"""Compare a repaired checkpoint with V2 on its same saved 200 test positions.

Reuses the original engine score cache. No new Stockfish analysis or training.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flychess.checkpoint import load_run
import chess
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--data",type=Path,default=Path("runs/fly-chess-v2/data"))
    parser.add_argument("--graph",type=Path,default=Path("data/fruitless/full-graph.npz"))
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    ns=load_run(args.run,args.data,args.graph,args.device)
    baseline=json.loads((args.baseline/"results.json").read_text())
    cache=json.loads((args.baseline/"engine-scores.json").read_text())
    positions=np.load(args.data/"fly-chess/positions-v2-full-1000000.npz")
    fens,labels,values=positions["fen"],positions["move"],positions["value"]
    selected=baseline["positions"]["main"]
    indices=np.array([int(k) for k in selected])
    predictions,correct,losses,scored_keys=[],0,[],[]
    for start in range(0,len(indices),32):
        batch=indices[start:start+32]
        boards=[chess.Board(str(fens[i])) for i in batch]
        logits,value=ns["predict"](ns["model"],boards)
        choices=ns["mask_logits"](logits,boards).argmax(1).cpu().numpy()
        predictions.extend(value.cpu().numpy().tolist())
        for i,board,choice in zip(batch,boards,choices):
            move=ns["index_to_move"](choice,board)
            correct+=int(move.uci()==str(labels[i]))
            scores={}
            for candidate in board.legal_moves:
                key="|".join((board.fen(en_passant="fen"),candidate.uci(),ns["STOCKFISH_BINARY_SHA"],"12"))
                if key not in cache:
                    break
                scores[candidate.uci()]=cache[key]
            else:
                losses.append(ns["position_loss"](scores,move))
                scored_keys.append(str(i))
    oldvalues=np.array([selected[str(i)]["decision"]["value"] for i in indices])
    old_mse=float(np.mean((oldvalues-values[indices])**2))
    new_mse=float(np.mean((np.array(predictions)-values[indices])**2))
    report={"n":len(indices),"test_split":"Same V2 saved position-level test sample",
            "v2_move_match":baseline["controls"]["main"]["legal_move_match"],"v3_move_match":correct/len(indices),
            "v2_value_mse":old_mse,"v3_value_mse":new_mse,
            "relative_value_error_reduction":1-new_mse/old_mse,
            "cp_score_cache_positions":len(losses),"note":"One seed; prediction metrics do not establish Elo"}
    def mean_cp(entries):
        ordinary=[e["cp_loss"] for e in entries if e["cp_loss"] is not None]
        return {"mean":float(np.mean(ordinary)) if ordinary else None,"n":len(ordinary)}
    report["v2_cp_loss"]=mean_cp([selected[k]["chosen"] for k in scored_keys])
    report["v3_cp_loss"]=mean_cp(losses)
    report["v2_expected_score_loss"]=float(np.mean([selected[k]["chosen"]["expected_score_loss"] for k in scored_keys])) if losses else None
    report["v3_expected_score_loss"]=float(np.mean([e["expected_score_loss"] for e in losses])) if losses else None
    out=args.run/"heldout-comparison.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
