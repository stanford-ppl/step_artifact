mv step-perf/src/memory/mod.rs step-perf/src/memory/bw_64_mod.rs
mv step-perf/src/memory/hdl_mod.rs step-perf/src/memory/mod.rs
source setup.sh
pytest hdl_validation/expert_tiling_sweep.py::test_expert_tiling_sweep_single_schedule
mv step-perf/src/memory/mod.rs step-perf/src/memory/hdl_mod.rs
mv step-perf/src/memory/bw_64_mod.rs step-perf/src/memory/mod.rs
source setup.sh
echo "figure 8 (1/2) done"
echo "figure 8 (STeP Simulator - Organge dots) done."
echo "IMPORTANT: The HDL numbers still have to be generated!"