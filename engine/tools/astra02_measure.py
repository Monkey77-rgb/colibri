#!/usr/bin/env python3
"""Dispatch-2 sequential harness; run inside a 22 GiB/no-swap systemd scope.
Keeps exact commands, per-process polled high-water RSS/read bytes, and device
sector deltas. Trace runs are training data, not throughput measurements.
No cache drops; pair zero is discarded for each timed comparison.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

ENGINE = Path(__file__).resolve().parents[1]
RAW = Path("/home/monkey/Documents/Ai/Hardware/diagnostics/host/2026-09-17-desktop-diskio")
MODEL = "/home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf"
LLAMA = "/home/monkey/src/llama.cpp-b9766/build/bin/llama-completion"
PROMPTS = {
    "train_network": "Write a short paragraph explaining how TCP detects missing packets and adjusts its transmission rate when a network becomes congested.",
    "train_plants": "Write a short paragraph explaining how plants convert sunlight into stored chemical energy, and why water availability matters.",
    "train_music": "Write a short paragraph explaining how rhythm and harmony create musical tension, and how composers resolve it.",
    "train_physics": "Write a short paragraph explaining why a satellite stays in orbit around Earth, and how its speed affects the orbit.",
    "eval1": "Write a short paragraph explaining how a B-tree keeps lookups fast on disk, and why node fan-out matters.",
    "eval2": "Write a short paragraph explaining why sourdough bread rises during proofing, and why dough hydration matters.",
    "io09": "Write a short paragraph explaining how a hash table resolves collisions with open addressing, and why load factor matters.",
}


def sectors():
    return int(next(l.split()[5] for l in Path("/proc/diskstats").read_text().splitlines()
                    if l.split()[2] == "nvme1n1"))


def run(label, prompt_name, profile=False, trace=False, llama=False, discard=False):
    output = RAW / f"astra02_{label}_raw.txt"
    prompt = RAW / f"astra02_prompt_{prompt_name}.txt"
    prompt.write_text(PROMPTS[prompt_name] + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("COLI_")}
    measured_env = dict(OMP_NUM_THREADS="8", OMP_WAIT_POLICY="active",
                        COLI_CPU_PROF="1", COLI_EXPERT_GB="12")
    if profile:
        measured_env["COLI_MOE_PROFILE"] = str(RAW / "astra02_profile_general.txt")
    if trace:
        tr = RAW / f"astra02_trace_{prompt_name}.txt"
        tr.open("x").close()  # trace appends; reject accidental reruns
        measured_env["COLI_MOE_TRACE"] = str(tr)
    env.update(measured_env)
    if llama:
        cmd = [LLAMA, "-m", MODEL, "-t", "8", "-ngl", "99", "-ncmoe", "32", "-c", "512",
               "-n", "96", "--temp", "0", "-p", PROMPTS[prompt_name], "-no-cnv"]
    else:
        cmd = [str(ENGINE / "coli-gpu"), MODEL, "--backend", "auto", "-p", PROMPTS[prompt_name],
               "-n", "96", "--temp", "0", "-c", "512"]
    cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
    cap = (cgroup / "memory.max").read_text().strip()
    swap = (cgroup / "memory.swap.max").read_text().strip()
    if cap != str(22*1024**3) or swap != "0":
        raise RuntimeError(f"wrong cgroup limits {cap}/{swap}")
    record = dict(label=label, prompt=prompt_name, discard=discard, command=cmd, env=measured_env,
                  time=datetime.datetime.now().astimezone().isoformat(), cap=cap, swap=swap,
                  load_before=Path("/proc/loadavg").read_text().strip(),
                  gpu_before=subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], text=True).strip())
    s0 = sectors(); t0 = time.monotonic(); hwm = rb = 0
    with output.open("x") as f:
        f.write(json.dumps(record) + "\n"); f.flush()
        process = subprocess.Popen(cmd, cwd=ENGINE, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while process.poll() is None:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                m = re.search(r"VmHWM:\s+(\d+)", status)
                if m: hwm = max(hwm, int(m[1]))
                io = Path(f"/proc/{process.pid}/io").read_text()
                rb = max(rb, int(re.search(r"^read_bytes:\s+(\d+)", io, re.M)[1]))
            except FileNotFoundError:
                pass
            time.sleep(0.25)
        record.update(exit=process.wait(), wall_s=time.monotonic()-t0, VmHWM_kB_polled=hwm,
                      proc_read_bytes_polled=rb, nvme_sectors=sectors()-s0,
                      load_after=Path("/proc/loadavg").read_text().strip())
    record["counters"] = [l for l in output.read_text(errors="replace").splitlines()
                          if re.search(r"prefill \d|generated \d|expert-calls|gpu expert slots|prefetch fills|prompt eval time|^.*eval time =|estore.*budget", l)]
    print(json.dumps(record), flush=True)
    if record["exit"]: raise RuntimeError(f"{label} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "eval1", "eval2", "paired"])
    stage = parser.parse_args().stage
    if stage == "train":
        for name in PROMPTS:
            if name.startswith("train_"): run(name, name, trace=True)
    elif stage.startswith("eval"):
        for pair in range(3):
            for arm in ["noprof", "prof"]:
                run(f"{stage}_{arm}_{pair}", stage, profile=arm=="prof", discard=pair==0)
    else:
        for pair in range(3):
            run(f"paired_prof_{pair}", "io09", profile=True, discard=pair==0)
            run(f"paired_llama_{pair}", "io09", llama=True, discard=pair==0)
