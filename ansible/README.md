# Running experiments on CCAD

CEFET-MG's HPC cluster (SLURM, shared NFS `$HOME`, CPU partition `short` with `short_qos`). Needs the CEFET VPN, which drops when idle, so
every run is submit-then-collect. Layout follows `dl2l/ansible`.

```sh
pip install ansible && ansible-galaxy collection install -r ansible/requirements.yml
echo 'CCAD_USERNAME=<cpf-no-punctuation>' > .env.local        # gitignored; or pass -e ccad_user=...
cd ansible
ansible-playbook -i inventories/ccad provision.yml                   # once: code + venv (python3.11) on the login node
ansible-playbook -i inventories/ccad submit.yml  -e spec=smoke       # one SLURM array job per spec; returns immediately
ansible-playbook -i inventories/ccad collect.yml -e spec=smoke       # status + rsync results into results/ (safe to repeat)
```

A spec (`experiments/<name>.yml`) lists `name`, `time`, `mem`, `max_parallel`, `results_dir` and `jobs`, one argument line per
array task (no `--results`; the template adds it). Optional keys: `command` (the console script each line is passed to, default
`snake-run`; e.g. `pacman-run`), and `ccad_partition` / `ccad_qos` to leave the default `short` partition (1 day). Per-user limits seen
on 2026-10-13: `normal` (`normal_qos`) 3 days and 96 CPUs, `extended` (`extended_qos`) 21 days and 48 CPUs. After the code gains a
console script, re-run `provision.yml` so the cluster venv installs it. Each task runs on one CPU with one BLAS thread and writes `done/<n>` with its exit code.
Compute nodes have no internet, so the code is rsynced from this machine and the venv is built on the login node. Logs are in
`~/snake/runs/<name>/logs/` on the cluster.

Specs: `smoke` (3 short jobs: venv, timing, memory), `e6_absolute_actions`, `nec_lr_sweep_absolute`.

Cluster runs reproduce local runs exactly for the same seed (checked on the 12 E6 runs). Measured on CCAD with 24 tasks running at once:
DQN on pixels took 1.7-2.0 h per 60k steps (7x7 and 10x10), NEC 1.0-1.7 h at 10x10.
