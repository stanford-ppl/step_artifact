# Figure 8 (2m)
cd /root/step_artifact/
source ./hdl_validation/figure8_step.sh  
# Produced file: step_artifact/hdl_validation/fig8.csv

# --------------------------------------------------------------------
# Figure 9 (60m, 90m)
cd /root/step_artifact/
pytest dyn_tiling/test_mixtral_sweep_revision.py::test_mixtral_b64
# Produced file: step_artifact/dyn_tiling/figure_9_mixtral_b64_raw.csv

pytest dyn_tiling/test_qwen_sweep_revision.py::test_qwen_b64_ablation # 
# Produced file: step_artifact/dyn_tiling/figure_9_qwen_b64_raw.csv

python dyn_tiling/generate_fig9_pareto_log.py
# Produced file: step_artifact/dyn_tiling/figure9.pdf

echo "figure 9 done"

# --------------------------------------------------------------------
# Figure 10 (380m, 650m)
cd /root/step_artifact/
pytest dyn_tiling/test_mixtral_sweep_prefill_revision.py::test_mixtral_b1024
# Produced file: step_artifact/dyn_tiling/figure_10_mixtral_b1024_raw.csv

pytest dyn_tiling/test_qwen_sweep_prefill_revision.py::test_qwen_b1024_ablation
# Produced file: step_artifact/dyn_tiling/figure_10_qwen_b1024_raw.csv

python dyn_tiling/generate_fig10_pareto_log.py
# Produced file: step_artifact/dyn_tiling/figure10.pdf

echo "figure 10 done"

# --------------------------------------------------------------------
# Figure 12 & 13 (48m, 52m)
cd /root/step_artifact/
pytest timeshare_mem_bound/test_membound_qwen_sweep_revet.py::test_static_tile
# Produced files: step_artifact/timeshare_mem_bound/fig_8_a.csv, 
#                 step_artifact/timeshare_mem_bound/fig_9_a.csv,
#                 step_artifact/timeshare_mem_bound/fig_9_b.csv

pytest timeshare_mem_bound/test_membound_qwen_sweep_dyn_tile.py::test_dyn_tile
# Produced files: step_artifact/timeshare_mem_bound/fig_8_b.csv

python timeshare_mem_bound/generate_fig12.py 
# Produced file: step_artifact/timeshare_mem_bound/figure12.pdf
python timeshare_mem_bound/generate_fig13.py 
# Produced file: step_artifact/timeshare_mem_bound/figure13.pdf


echo "figure 12 & 13 done"
# --------------------------------------------------------------------
# Figure 21 (1m39s, 4m50s, 6m31s)
cd /root/step_artifact/
pytest dynamic_par/sweep_ae.py::test_b16_sweep
# Produced file: step_artifact/dynamic_par/batch16_sweep_ae.csv

pytest dynamic_par/sweep_ae.py::test_b64_sweep
# Produced file: step_artifact/dynamic_par/batch64_sweep_ae.csv

pytest dynamic_par/sweep_ae.py::test_b64_b16_sweep
# Produced file: step_artifact/dynamic_par/batch80_sweep_ae.csv

python dynamic_par/fig21_change_scale.py
# Produced file: step_artifact/dynamic_par/figure21.pdf
echo "figure 21 done"


# --------------------------------------------------------------------
echo "figure 14 start"

pytest dynamic_par/sweep_ae_revision.py::test_b64_sweep # 3m30s
# Produced file: step_artifact/dynamic_par/batch64_interleave_dynamic.csv

python dynamic_par/fig_interleave_dyn.py
# Produced file: step_artifact/dynamic_par/figure14.pdf

echo "figure 14 done"
# --------------------------------------------------------------------
echo "figure 15 start"

pytest dynamic_par/sweep_ae_revision.py::test_batch_sweep # 1m
# Produced file: step_artifact/dynamic_par/batch_sweep_coarse_vs_dynamic.csv

python dynamic_par/fig_coarse_dyn_64.py
# Produced file: step_artifact/dynamic_par/figure15.pdf

echo "figure 15 done"


# --------------------------------------------------------------------