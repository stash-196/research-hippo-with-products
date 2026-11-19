"""Run the 2026-11-19 window checks into a NEW timestamped folder:
    outputs/freq_dev/window_checks/<YYYYmmdd_HHMMSS>/
Usage (from repo root):  python -m lag_op_hippo.window_checks.run_window_checks [step0 A B C]
Never writes into outputs/freq_dev/latest or outputs/freq_dev/runs.

Labels saying "repo ..." mean "the lag_op_hippo library at the commit recorded in
run_config.json". Up to tag pre-libfix-2026-11-19 (e.g. run 20261119_172639) that is
the unfixed library; after the 2026-11-19 library fix the "repo" routes ARE the fixed
builders, so compare against 20261119_172639 for the before values."""
import datetime, json, os, runpy, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STEPS = {"step0": "step0_legt_fout_fix", "A": "exp_a_per_window", "B": "exp_b_two_window", "C": "exp_c_eigen"}


def main(argv):
    steps = argv or list(STEPS)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPO, "outputs", "freq_dev", "window_checks", ts)
    assert not os.path.exists(out)
    os.makedirs(out)
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    cfg = dict(timestamp=ts, git_commit=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
               git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")), steps=steps, python=sys.version,
               hyperparameters=dict(alpha=0.2, window_length=1.0, eps=1e-8, N=16, N_domain_expansion=64, dt=1e-3,
                                    T_step0=3.0, T_A=4.0, T_B=3.0, quad_order="200*N", discretization="bilinear",
                                    signals={"cos": "cos(2 pi 3 s) (repo 02.01/02.02 input)",
                                             "chirp": "sin(2 pi (0.5 s + 0.35 s^2))"}))
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    os.environ["WINDOW_CHECKS_OUT"] = out
    for s in steps:
        print(f"=== {s} ===", flush=True)
        runpy.run_module(f"lag_op_hippo.window_checks.{STEPS[s]}", run_name="__main__")
    print("output:", out)


if __name__ == "__main__":
    main(sys.argv[1:])
