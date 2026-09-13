"""Closed-loop adapter for fly-wirehead's full MaleCNS v1.0 brain.

The wiring, neuron model, kernel and retinal projection are fly-wirehead's,
imported without modification. This module only chooses which annotated cells
receive sensory current and which cells are read out. Those choices are
modelling assumptions; `describe()` records exactly which cells were used.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIREHEAD = ROOT / "experiments" / "fly-wirehead"
WIREHEAD_DATA = ROOT / "data" / "fly-wirehead"
RUNS = ROOT / "runs" / "fly-heaven"

TICK_MS = 50.0
SILENCE_MV = -40.0
FMA_FLAGS = ["-O3", "-mfma", "-ffp-contract=fast", "-std=c++17", "-shared", "-fPIC"]
# experiments/fruitless/experiment/followup/bounded_routes.py:45
P1_BODY_IDS = [12442, 16719, 17867, 20117, 20803, 23968, 519518, 522419]
VAB3_BODY_IDS = [11998, 13341, 13693, 512498]

SENSES = {
    "sugar": "LB3c labellar gustatory neurons (fly-wirehead's `sugar` population)",
    "antenna": "JO-C and JO-E Johnston's organ neurons",
    "loom": "LC4 visual projection neurons",
    "heat": "TRN_VP2 thermosensory neurons",
    "cold": "TRN_VP3a/b thermosensory neurons",
    "female": "LgLG5/LgLG8 foreleg gustatory neurons (ProLN): candidate female-cue cells",
    "male": "LgLG6/LgLG7 foreleg gustatory neurons (ProLN): candidate male-cue cells, calibration control",
}
READOUTS = {
    "MN9": "MN9 proboscis motor neurons",
    "DNg12": "DNg12 grooming descending neurons",
    "GF": "DNp01 giant fiber descending neurons (escape takeoff)",
    "P1": "Eight literature-mapped P1 cells, pC1_4a/b (fruitless follow-up)",
    "pIP10": "pIP10 courtship-song descending neurons",
    "vPR6": "vPR6 song-pattern neurons",
    "DNa02_L": "DNa02 steering descending neuron, left soma",
    "DNa02_R": "DNa02 steering descending neuron, right soma",
    "MDN": "Moonwalker descending neurons (backward walking)",
    "motor": "All VNC motor neurons",
    "PAM11": "PAM11 dopamine neurons",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cpu_has_fma():
    info = Path("/proc/cpuinfo")
    return info.exists() and any(
        "fma" in line.split() for line in info.read_text().splitlines() if line.startswith("flags")
    )


def _load_wirehead():
    if not (WIREHEAD_DATA / "graph.npz").exists():
        raise FileNotFoundError("Prepared MaleCNS graph missing: run fly-heaven.ipynb sections 0 and 2.1 first.")
    os.environ["FLYWIREHEAD_DATA"] = str(WIREHEAD_DATA)
    if str(WIREHEAD) not in sys.path:
        sys.path.insert(0, str(WIREHEAD))
    import flywirehead.neural.brain as wirehead_brain
    from flywirehead.neural.common import annotations
    from flywirehead.neural.visual import VisualMemoryBrain

    # fly-heaven.ipynb §2.1: fly-wirehead's published arithmetic needs fused multiply-adds on x86-64.
    if platform.machine() == "x86_64" and _cpu_has_fma():
        library = WIREHEAD_DATA / "cache/physiology-v6-fma/libmemory.so"
        record_path = library.with_suffix(".so.json")
        source_sha = sha256(wirehead_brain.SOURCE)
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        if record.get("source_sha256") != source_sha or record.get("flags") != FMA_FLAGS or not library.exists():
            library.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["c++", *FMA_FLAGS, str(wirehead_brain.SOURCE), "-o", str(library)], check=True)
            record = {"model": wirehead_brain.MODEL, "source_sha256": source_sha, "binary_sha256": sha256(library), "flags": FMA_FLAGS}
            record_path.write_text(json.dumps(record, indent=2))
        wirehead_brain.LIBRARY = library
    return VisualMemoryBrain, annotations


class ConnectomeBrain:
    """The full 166,700-neuron graph, advanced 50 ms at a time with sensory currents."""

    def __init__(self, silence=()):
        VisualMemoryBrain, annotations = _load_wirehead()
        b = self.brain = VisualMemoryBrain()
        b.weights_frozen = True  # no KC→MBON learning: restores and replays stay exact
        a = annotations(b.ids)
        types = a.type.fillna("")
        self.types = types.to_numpy()
        self.body = a.index.to_numpy()
        side = a.somaSide.fillna("").to_numpy()
        foreleg = a.entryNerve.fillna("").to_numpy() == "ProLN"

        def pick(mask):
            return np.flatnonzero(np.asarray(mask)).astype(np.int32)

        self.senses = {
            "sugar": b.sugar,
            "antenna": pick(types.str.startswith("JO-C") | types.str.startswith("JO-E")),
            "loom": pick(types.eq("LC4")),
            "heat": pick(types.eq("TRN_VP2")),
            "cold": pick(types.str.startswith("TRN_VP3")),
            "female": pick(types.isin(["LgLG5", "LgLG8"]).to_numpy() & foreleg),
            "male": pick(types.isin(["LgLG6", "LgLG7"]).to_numpy() & foreleg),
        }
        dna02 = types.eq("DNa02").to_numpy()
        self.readouts = {
            "MN9": pick(types.eq("MN9")),
            "DNg12": pick(types.str.startswith("DNg12")),
            "GF": pick(types.eq("DNp01")),
            "P1": pick(np.isin(self.body, P1_BODY_IDS)),
            "pIP10": pick(types.eq("pIP10")),
            "vPR6": pick(types.eq("vPR6")),
            "DNa02_L": pick(dna02 & (side == "L")),
            "DNa02_R": pick(dna02 & (side == "R")),
            "MDN": pick(types.eq("MDN")),
            "motor": pick(a.superclass.eq("vnc_motor").to_numpy()),
            "PAM11": b.circuit["reward"],
        }
        empty = [k for k, ix in {**self.senses, **self.readouts}.items() if not len(ix)]
        if empty or len(self.readouts["P1"]) != len(P1_BODY_IDS):
            raise ValueError(f"Missing annotated cells: {empty or 'P1'}")

        # Glutamatergic relays treated as excitatory, exactly as fruitless bounded_routes.py:48-49.
        self.sign_overrides = {"female": self.senses["female"], "vAB3": pick(np.isin(self.body, VAB3_BODY_IDS))}
        for cells in self.sign_overrides.values():
            for i in cells:
                edges = slice(b.ptr[i], b.ptr[i + 1])
                b.weight[edges] = np.abs(b.weight[edges])

        unknown = set(silence) - set(self.readouts)
        if unknown:
            raise ValueError(f"Unknown readouts to silence: {sorted(unknown)}")
        self.silenced = {k: self.readouts[k] for k in silence}

    def step(self, frame, currents, ms=TICK_MS):
        """Advance the brain; `currents` maps sense name to mV-equivalent current."""
        # `currents` may carry extra bookkeeping fields (heaven.world.World also
        # reports things like near_fruit/female_distance for the behaviour layer
        # to read); only known sense names are ever turned into brain current.
        pulses = [(self.senses[k], float(v)) for k, v in currents.items() if k in self.senses and v]
        pulses += [(ix, SILENCE_MV) for ix in self.silenced.values()]
        counts, _ = self.brain.rgb_step(frame, ms, learning=False, stimulation=pulses or None)
        seconds = ms / 1000
        rates = {k: float(counts[ix].sum() / (len(ix) * seconds)) for k, ix in self.readouts.items()}
        rates["network_spikes"] = int(counts.sum())
        return rates

    def snapshot(self):
        b = self.brain
        return {"fields": {k: getattr(b, k).copy() for k in b.fields}, "cursor": b.cursor, "total_spikes": b.total_spikes}

    def restore(self, state):
        # Weights are frozen, so the dynamic fields are the whole state.
        b = self.brain
        for k, v in state["fields"].items():
            getattr(b, k)[...] = v
        b.cursor = state["cursor"]
        b.sim_ms = b.cursor * b.dt
        b.total_spikes = state["total_spikes"]

    def describe(self):
        def cells(ix, label):
            return {"cells": label, "count": int(len(ix)), "types": sorted(set(self.types[ix])), "body_ids": [str(i) for i in self.body[ix]]}

        return {
            "neurons": int(self.brain.n),
            "connections": int(len(self.brain.post)),
            "kernel": self.brain.build,
            "senses": {k: cells(ix, SENSES[k]) for k, ix in self.senses.items()},
            "readouts": {k: cells(ix, READOUTS[k]) for k, ix in self.readouts.items()},
            "sign_overrides": {k: [str(i) for i in self.body[ix]] for k, ix in self.sign_overrides.items()},
            "silenced": sorted(self.silenced),
            "silence_mV": SILENCE_MV,
        }
