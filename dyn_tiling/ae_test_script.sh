
# --------------------------------------------------------------------
# Figure 9 
date
time pytest dyn_tiling/test_mixtral_sweep_revision.py::test_mixtral_b64 -s # 60m
# Produced file: step_artifact/dyn_tiling/figure_9_mixtral_b64_raw.csv

date
time pytest dyn_tiling/test_qwen_sweep_revision.py::test_qwen_b64_ablation -s # 90m
# Produced file: step_artifact/dyn_tiling/figure_9_qwen_b64_raw.csv


date
python dyn_tiling/generate_fig9_pareto_log.py
# Produced file: step_artifact/dyn_tiling/figure9.pdf

echo "figure 9 done"

# --------------------------------------------------------------------
# Figure 10
date
time pytest dyn_tiling/test_mixtral_sweep_prefill_revision.py::test_mixtral_b1024 -s # 380m
# Produced file: step_artifact/dyn_tiling/figure_10_mixtral_b1024_raw.csv

date
time pytest dyn_tiling/test_qwen_sweep_prefill_revision.py::test_qwen_b1024_ablation -s # 650m
# Produced file: step_artifact/dyn_tiling/figure_10_qwen_b1024_raw.csv

date
python dyn_tiling/generate_fig10_pareto_log.py
# Produced file: step_artifact/dyn_tiling/figure10.pdf

echo "figure 10 done"