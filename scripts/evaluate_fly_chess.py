"""Resume a saved V2–V5.5 model's strength benchmark without retraining.

Run from the repository root: .venv/bin/python scripts/evaluate_fly_chess.py --help
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flychess.checkpoint import load_run
from flychess.strength import strength_summary
import chess
import chess.engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/fly-chess-v2/data"))
    parser.add_argument("--graph", type=Path, default=Path("data/fruitless/full-graph.npz"))
    parser.add_argument("--engine", type=Path, default=Path("data/fly-chess/stockfish/stockfish"))
    parser.add_argument("--out", type=Path, help="Separate benchmark folder for another clock/configuration")
    parser.add_argument("--checkpoint", default="main.pt", choices=("main.pt","main.best.pt","main.champion.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=20, help="Number of varied opening pairs (2 games each)")
    parser.add_argument("--minutes", type=float, default=30, help="Fresh time allowance on every invocation")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--reference-elo", type=int, default=None, help="Stockfish UCI_Elo; default uses its supported minimum")
    parser.add_argument("--clock", type=float, default=120)
    parser.add_argument("--increment", type=float, default=1)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 1000 or args.minutes <= 0 or args.clock <= 0 or args.increment < 0:
        parser.error("Use 1–1000 opening pairs, a positive allowance/clock, and a nonnegative increment")
    ns = load_run(args.run, args.data, args.graph, args.device, checkpoint_name=args.checkpoint)
    # Execute only the opening constants/validation, never the experiment loop.
    notebook = json.loads((Path(__file__).resolve().parents[1] / "fly_chess_colab_V3.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if "matches" in c["metadata"].get("tags", []))
    source = "".join(cell["source"])
    exec(source[source.index("OPENINGS = ["):source.index("def model_opponent")], ns)
    import ast
    opening_definition = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "make_opening_pairs")
    exec(compile(ast.Module(body=[opening_definition], type_ignores=[]), "openings", "exec"), ns)
    openings = ns["make_opening_pairs"](args.pairs)
    binary = args.engine
    with binary.open("rb") as handle:
        engine_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if engine_hash != ns["STOCKFISH_BINARY_SHA"]:
        raise ValueError("Stockfish binary differs from the saved run; pass its matching --engine")
    with (args.run / args.checkpoint).open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    config = {"checkpoint_sha256": checkpoint_hash,
              "search": args.search, "clock": args.clock, "increment": args.increment,
              "device": args.device, "engine_sha256": ns["STOCKFISH_BINARY_SHA"],
              "assist_mode": ns["manifest"].get("assistance", {}).get("source", "none")}
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:
        minimum, maximum = engine.options["UCI_Elo"].min, engine.options["UCI_Elo"].max
        rating = minimum if args.reference_elo is None else args.reference_elo
        if not minimum <= rating <= maximum:
            parser.error(f"Stockfish supports UCI_Elo {minimum}–{maximum}")
        config["reference_elo"] = rating
        engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": rating})
        out = args.out or args.run / "strength-benchmark" / ("search" if args.search else "policy") / f"sf-{rating}"
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Benchmark configuration changed; use a separate output folder")
        ns["atomic_json"](config_path, config)
        ns["RUN_DIR"] = out
        records_path = out / "records.json"
        records = json.loads(records_path.read_text()) if records_path.exists() else []
        deadline = time.monotonic() + args.minutes * 60
        player = ns["model_opponent"](ns["model"], args.search)
        def opponent(board, clock, increment):
            return engine.play(board, chess.engine.Limit(
                white_clock=clock if board.turn else args.clock,
                black_clock=clock if not board.turn else args.clock,
                white_inc=increment, black_inc=increment)).move
        try:
            for pair, opening in enumerate(openings):
                for color in ("white", "black"):
                    if time.monotonic() >= deadline:
                        return
                    game_id = f"{pair}-{color}"
                    if any(r["id"] == game_id for r in records):
                        continue
                    white, black = (player, opponent) if color == "white" else (opponent, player)
                    game = ns["play_game"](white, black, opening, game_id,
                                          initial_clock=args.clock, increment=args.increment)
                    score = game["score"]
                    if color == "black" and score is not None:
                        score = 1 - score
                    records.append({"id": game_id, "pair": pair, "color": color, "score": score, "reason": game["reason"]})
                    ns["atomic_json"](records_path, records)
                    print(game_id, score, game["reason"], flush=True)
        finally:
            summary = strength_summary(records, rating, "Stockfish UCI_Elo setting at the recorded time control")
            summary["benchmark_status"] = "complete" if len(records) >= args.pairs*2 else "paused"
            summary["target_pairs"] = args.pairs
            ns["atomic_json"](out / "summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
