"""One-dimensional mass-preserving Barenblatt benchmark on [0, 3].

Install with python -m pip install -r requirements.txt.
Run python run.py 3 to load and check the archived model for m=3.
Run python run.py 3 --train-to new_m3.npz to solve from the prescribed datum.
Supported exponents are 1.5, 3, 6, and 9.
"""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from root_fractional_mass import RootFractionalMassELM
from solver import Config

HERE = Path(__file__).resolve().parent
T = 3.0
EXPONENTS = (1.5, 3.0, 6.0, 9.0)


def initialize(m):
    baseline = Config(max_iter=80, step_tol=0.0)
    boundary_features = RootFractionalMassELM(m, baseline)
    config = Config(pressure_neurons=384, rcond=0.0, max_iter=80, step_tol=0.0)
    model = RootFractionalMassELM(m, config, order=128, scan=257, endpoint_weight="all")
    rng = np.random.default_rng(config.seed)
    model.W_v[:, 0] = rng.uniform(-np.pi / 2, np.pi / 2, model.W_v.shape)[:, 0]
    model.W_v[:, 1] *= 2.0
    model.W_s = boundary_features.W_s.copy()
    model.b_s = boundary_features.b_s.copy()
    model.slab_length = T
    model.initialize_pressure(T)
    return model


class PhaseStagnation(Exception):
    """Stop after five consecutive accepted relative decreases at most 1e-6."""


def stop_on_plateau(model, row):
    recent = model.history[-5:]
    if len(recent) == 5 and all(
        record.get("step", 0.0) > 0.0
        and 0.0 <= (record["loss"] - record["accepted_loss"]) / record["loss"] <= 1e-6
        for record in recent
    ):
        row["status"] = "residual_stagnation"
        raise PhaseStagnation


def train(m, output):
    if output.exists():
        raise FileExistsError(output)
    model = initialize(m)
    for policy in ("residual_continuous", "fixed"):
        model.history = []
        try:
            model.train([T], svd_policy=policy, column_scaling=False,
                        iteration_callback=stop_on_plateau)
        except PhaseStagnation:
            pass
        print(f"{policy}: {model.history[-1]['status']}")
    with output.open("xb") as file:
        model.save(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m", type=float, choices=EXPONENTS)
    parser.add_argument("--train-to", type=Path, help="Write a fresh solve to this NPZ file.")
    args = parser.parse_args()
    with threadpool_limits(4):
        if args.train_to:
            train(args.m, args.train_to)
        else:
            model = RootFractionalMassELM.load(HERE / "models" / f"m{args.m:g}.npz")
            print(f"m={model.m:g}, T={model.slab_length:g}, coefficients={model.nparams}")
            print(f"Native relative mass error at T=3: {abs(model.mass_curve([T])[0]/model.M0-1):.3e}")


if __name__ == "__main__":
    main()
