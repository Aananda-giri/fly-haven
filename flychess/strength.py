"""Score-based strength estimates with explicit reference scales.

Opening pairs, rather than individual games, are the independent sampling unit.
The bounded-score Hoeffding interval stays nonzero after an all-loss/all-win run.
It quantifies match sampling only, not uncertainty in an opponent's calibration.
"""
import math


def score_to_elo(score):
    if score <= 0 or score >= 1:
        return None
    return 400 * math.log10(score / (1 - score))


def strength_summary(records, reference_rating=None, reference_scale=None, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    if reference_rating is not None and not reference_scale:
        raise ValueError("A reference rating requires an explicit scale/source")
    if reference_rating is not None and (not isinstance(reference_rating,(int,float)) or not math.isfinite(reference_rating)):
        raise ValueError("Reference rating must be finite")
    groups = {}
    for record in records:
        score = record.get("score")
        if score is not None and score not in (0, 0.5, 1):
            raise ValueError("Game score must be 0, 0.5, 1, or unresolved")
        groups.setdefault(record["pair"], []).append(record)
    pairs = []
    for group in groups.values():
        if len(group) == 2 and {r["color"] for r in group} == {"white", "black"}:
            if all(r["score"] is not None for r in group):
                pairs.append(sum(r["score"] for r in group) / 2)
    report = {"complete_pairs": len(pairs), "games": len(records),
              "wins": sum(r.get("score") == 1 for r in records),
              "draws": sum(r.get("score") == .5 for r in records),
              "losses": sum(r.get("score") == 0 for r in records),
              "unresolved": sum(r.get("score") is None for r in records),
              "reference_rating": reference_rating, "reference_scale": reference_scale,
              "interval_method": "95% bounded-score Hoeffding over opening pairs" if alpha == .05
                                 else f"{1-alpha:.1%} bounded-score Hoeffding over opening pairs",
              "sampling_assumption": "Independent, representative opening pairs",
              "score": None, "score_interval": None, "elo_difference": None,
              "elo_difference_interval": None, "reference_scale_estimate": None,
              "reference_scale_interval": None, "status": "insufficient_games"}
    if not pairs:
        return report
    score = sum(pairs) / len(pairs)
    radius = math.sqrt(math.log(2 / alpha) / (2 * len(pairs)))
    interval = [max(0, score - radius), min(1, score + radius)]
    delta = score_to_elo(score)
    bounds = [score_to_elo(p) for p in interval]
    report.update(score=score, score_interval=interval, elo_difference=delta,
                  elo_difference_interval=bounds,
                  status="estimate" if delta is not None else "upper_bound" if score == 0 else "lower_bound")
    if interval == [0,1]:
        report["status"] = "unbounded_interval"
    if reference_rating is not None:
        report["reference_scale_estimate"] = reference_rating + delta if delta is not None else None
        report["reference_scale_interval"] = [reference_rating + b if b is not None else None for b in bounds]
        report["calibration_note"] = "Conditional on the reference's calibration; not a FIDE/Chess.com/Lichess rating"
    report["unbounded_endpoints"] = {"lower": interval[0] == 0, "upper": interval[1] == 1}
    return report
