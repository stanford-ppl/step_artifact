# STeP Artifact Evaluation

This is a repository for STeP artifact generation.

## Overview

* [Getting Started (5 human-minutes + 10 compute-minutes)](#getting-started-5-human-minutes--10-compute-minutes)
* [Run Experiments (5 human-minutes + 7 compute-hour)](#run-experiments-5-human-minutes--7-compute-hour)
* [Validate All Results](#validate-all-resultsvalidate)
* [[Optional] Detailed Explanation of What the Top-Level Script Does](#optional-detailed-explanation-of-what-the-top-level-script-does)
* [[Optional] To customise or extend the toolchain](#optional-to-customise-or-extend-the-toolchain)

## Getting Started (5 human-minutes + 10 compute-minutes)

This guide assumes the user has a working installation of Docker, git, and some version of Python 3 installed.

* Run the following commands to clone this repository and the [step-artifact-hdl](https://github.com/stanford-ppl/step-artifact-hdl) repository to the local machine.

    ```bash
    git clone --recursive https://github.com/stanford-ppl/step_artifact.git
    git clone https://github.com/stanford-ppl/step-artifact-hdl.git
    ```

* Build the Docker image with the following commands (the build can take upto 5 minutes)

    ```
    docker build -f step_artifact/Dockerfile -t step_artifact .
    ```

* The Docker container can be started with the following command. This will print the `CONTAINER_ID`.

    ```
    docker run -dit step_artifact bash
    ```

* The container can be attached to by running the command below using the `CONTAINER_ID` the previous step.

    ```
    docker attach <CONTAINER_ID>
    ```

  * IMPORTANT: Do not type `exit` in the docker terminal as this will stop the container. The proper way to detach the docker is the pressing sequence `CTRL+p`, `CTRL+q`.

* Run the following command to set up the environment. The following command has to run whenever a new terminal is opened.

    ```bash
    cd /root/step_artifact
    source setup.sh
    ```

## Run Experiments (5 human-minutes + 24.5 compute-hour)

All the experiments and figures can be run by the following commands. In total, it takes around 24.5 hours when tested on a machine with 8 vCPUs.

```bash
### In Docker Container ###
$ cd /root/step_artifact
# Figure 9,10,12,13,14,15,21 and half of Figure 8 (23 hr)
$ source ae_cmd.sh
# Figure 8 (1hr 30mins)
$ cp /root/step_artifact/hdl_validation/fig8.csv /root/step-artifact-hdl/step_reference.csv
$ cd /root/step-artifact-hdl
$ ./run_dse_and_figure.sh
```

Once all the experiments complete, detach the container by pressing `CTRL+p` and then `CTRL+q`. You can extract the tables/figures from the Docker container by following the instructions in the section [Validate All Results](#validate-all-results) in this README.

## Validate All Results

1. Exit the docker (CTRL+p, CTRL+q)
2. Move into the cloned `step_artifact` repository on the local machine and run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

    ```bash
    ### In the local machine ###
    $ cd step_artifact
    $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
    $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
    ```

    * `copy_from_docker.py` runs a series of docker cp commands to pull the figures from the container.
    * `--output_dir` is used to specify an output directory on the local machine for the figures to be stored in. The `mkdir -p <OUTPUT_DIRECTORY>` command will create the directory if it doesn't exist. The files referenced in the next few steps will be found at this directory.
    * `--docker_id` is used to identify the docker container ID. This should have printed when the docker was created and is the same ID used to attach to the container. You may also retrieve the CONTAINER_ID again by running `docker ps` in your terminal.
3. The expected results in the `step_artifact/<OUTPUT_DIRECTORY>` are:

    ```
    step_artifact/<OUTPUT_DIRECTORY>
    |_ step-artifact-hdl
    |_step_artifact
        |_dyn_tiling
        |_dynamic_par
        |_timeshare_mem_bound
    ```

    * Figure 8: The reproduced figure and experiment results can be found in the `step-artifact-hdl` folder. The `validation.pdf` should match Figure 8 in the paper. The values used to create the plot are in the other two CSV files in the `step-artifact-hdl` folder.

    * Figure 9: The reproduced figure and experiment results can be found in the `dyn_tiling` folder. The file `figure9.pdf` should match Figure 9 in the paper. The values used for creating the plot can be found in`figure_9_mixtral_b64_raw.csv` and `figure_9_qwen_b64_raw.csv`.

    * Figure 10: The reproduced figure and experiment results can be found in the `dyn_tiling` folder.  The file `figure10.pdf` should match Figure 10 in the paper. The values used for creating the plot can be found in `figure_10_mixtral_b1024_raw.csv` and `figure_10_qwen_b1024_raw.csv`.

    * Figure 12: The reproduced figure and experiment results can be found in the `timeshare_mem_bound` folder. The file `figure12.pdf` should match Figure 12 in the paper. The values used to create the plot are in `fig_8_a.csv` and `fig_8_b.csv`.

    * Figure 13: The reproduced figure and experiment results can be found in the `timeshare_mem_bound` folder. The file `figure13.pdf` should match Figure 13 in the paper. The values used to create the plot are in `fig_9_a.csv` and `fig_9_b.csv`.

    * Figure 14: The reproduced figure and experiment results can be found in the `dynamic_par` folder. The file `figure14.pdf` should match Figure 14 in the paper. The values used for creating the plot can be found in `batch64_interleave_dynamic.csv`.

    * Figure 15: The reproduced figure and experiment results can be found in the `dynamic_par` folder. The file `figure15.pdf` should match Figure 15 in the paper. The values used for creating the plot can be found in `batch_sweep_coarse_vs_dynamic.csv`.

    * Figure 21: The reproduced figure and experiment results can be found in the `dynamic_par` folder. The file `figure21.pdf` should match Figure 21 in the paper. The values used to generate the plot are provided in the remaining CSV files in the same directory.

## [Optional] Detailed Explanation of What the Top-Level Script Does

### Run and Validate Figure 8 (10 human-minutes + 2 compute-hours)

* Run the following commands:
    1. Generates the STeP Simulator numbers (organe dots) in Figure 5. The numbers will be stored in `/root/step_artifact/hdl_validation/fig8.csv`.

        ```bash
        ### In the docker container ###
        $ cd /root/step_artifact/
        $ source ./hdl_validation/figure8_step.sh    
        ```

    2. Run the HDL simulation, copy the results from STeP simulator to the designated location, and generate figure 8.

        ```bash
        # Copy the simulation resuls for the STeP simulator (fig8.csv) to the designated location to generate the graph
        $ cp /root/step_artifact/hdl_validation/fig8.csv /root/step-artifact-hdl/step_reference.csv
        # Run the HDL simulation and generate the figure
        $ cd /root/step-artifact-hdl
        $ ./run_dse_and_figure.sh
        ```

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. As there will only be results related to figure 8 generated, modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 5 as follows:

        ```python
        FILES_TO_COPY = [
            "step-artifact-hdl/dse_results.csv",
            "step-artifact-hdl/step_reference.csv",
            "step-artifact-hdl/validation.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine (step_artifact repository) ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `step-artifact-hdl` folder. The `validation.pdf` should match Figure 8 in the paper. The values used to create the plot are in the other two CSV files in the `step-artifact-hdl` folder.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_ step-artifact-hdl
            |_ dse_results.csv
            |_ step_reference.csv
            |_ validation.pdf

        ```

### Run and Validate Figure 9 (5 human-minutes + 150 compute-minutes)

* Run the following commands

    ```bash
    ### In the docker container ###
    cd /root/step_artifact/
    source setup.sh

    pytest dyn_tiling/test_mixtral_sweep_revision.py::test_mixtral_b64 # 60m
    # Produced file: step_artifact/dyn_tiling/figure_9_mixtral_b64_raw.csv

    pytest dyn_tiling/test_qwen_sweep_revision.py::test_qwen_b64_ablation # 90m
    # Produced file: step_artifact/dyn_tiling/figure_9_qwen_b64_raw.csv

    python dyn_tiling/generate_fig9_pareto_log.py
    # Produced file: step_artifact/dyn_tiling/figure9.pdf

    echo "figure 9 done"
    ```

  * The `test_mixtral_b64` will run the left portion of figure 9 (Mixtral8x7B) and produce `step_artifact/dyn_tiling/figure_9_mixtral_b64_raw.csv`.
  * The `test_qwen_b64_ablation` will run the right portion of figure 9 (Qwen3-30B-A3B) and produce `step_artifact/dyn_tiling/figure_9_qwen_b64_raw.csv`.

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 9 as follows:

        ```python
        FILES_TO_COPY = [
            "step_artifact/dyn_tiling/figure_9_mixtral_b64_raw.csv",
            "step_artifact/dyn_tiling/figure_9_qwen_b64_raw.csv",
            "step_artifact/dyn_tiling/figure9.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `dyn_tiling` folder. The file `figure9.pdf` should match Figure 9 in the paper. The values used for creating the plot can be found in`figure_9_mixtral_b64.csv` and `figure_9_qwen_b64.csv`.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_step_artifact
            |_dyn_tiling
                |_ figure_9_mixtral_b64_raw.csv
                |_ figure_9_qwen_b64_raw.csv
                |_ figure9.pdf
        ```

### Run and Validate Figure 10 (5 human-minutes + 17 compute-hours)

* Run the following commands

    ```bash
    ### In the docker container ###
    cd /root/step_artifact/
    pytest dyn_tiling/test_mixtral_sweep_prefill_revision.py::test_mixtral_b1024 
    # Produced file: step_artifact/dyn_tiling/figure_10_mixtral_b1024_raw.csv

    pytest dyn_tiling/test_qwen_sweep_prefill_revision.py::test_qwen_b1024_ablation
    # Produced file: step_artifact/dyn_tiling/figure_10_qwen_b1024_raw.csv

    python dyn_tiling/generate_fig10_pareto_log.py
    # Produced file: step_artifact/dyn_tiling/figure10.pdf

    echo "figure 10 done"
    ```

  * The `test_mixtral_b1024` will run the left portion of figure 10 (Mixtral8x7B) and produce `step_artifact/dyn_tiling/figure_10_mixtral_b1024_raw.csv`.
  * The `test_qwen_b1024_ablation` will run the right portion of figure 10 (Qwen3-30B-A3B) and produce `step_artifact/dyn_tiling/figure_10_qwen_b1024_raw.csv`.

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 10 as follows:

        ```python
        FILES_TO_COPY = [
            "step_artifact/dyn_tiling/figure_10_mixtral_b1024_raw.csv",
            "step_artifact/dyn_tiling/figure_10_qwen_b1024_raw.csv",
            "step_artifact/dyn_tiling/figure10.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `dyn_tiling` folder.  The file `figure10.pdf` should match Figure 10 in the paper. The values used for creating the plot can be found in `figure_10_mixtral_b1024_raw.csv` and `figure_10_qwen_b1024_raw.csv`.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_step_artifact
            |_dyn_tiling
                |_ figure_10_mixtral_b1024_raw.csv
                |_ figure_10_qwen_b1024_raw.csv
                |_ figure10.pdf
        ```

### Run and Validate Figure 12 (5 human-minutes + 100 compute-minutes)

* Run the following commands

    ```bash
    ### In the docker container ###
    cd /root/step_artifact/
    source setup.sh
    pytest timeshare_mem_bound/test_membound_qwen_sweep_revet.py::test_static_tile
    # Produced files: step_artifact/timeshare_mem_bound/fig_8_a.csv, 
    #                 step_artifact/timeshare_mem_bound/fig_9_a.csv,
    #                 step_artifact/timeshare_mem_bound/fig_9_b.csv

    pytest timeshare_mem_bound/test_membound_qwen_sweep_dyn_tile.py::test_dyn_tile
    # Produced files: step_artifact/timeshare_mem_bound/fig_8_b.csv

    python timeshare_mem_bound/generate_fig12.py 
    # Produced file: step_artifact/timeshare_mem_bound/figure12.pdf
    ```

  * The `test_static_tile` will run experiments for figure 12(a) and produce `step_artifact/timeshare_mem_bound/fig_8_a.csv`.
  * The `test_dyn_tile` will run experiments for figure 12(b) and produce `step_artifact/timeshare_mem_bound/fig_8_b.csv`.

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 12 as follows:

        ```python
        FILES_TO_COPY = [
            "step_artifact/timeshare_mem_bound/fig_8_a.csv",
            "step_artifact/timeshare_mem_bound/fig_8_b.csv",
            "step_artifact/timeshare_mem_bound/figure12.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `timeshare_mem_bound` folder. The file `figure12.pdf` should match Figure 8 in the paper. The values used to create the plot are in `fig_8_a.csv` and `fig_8_b.csv`.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_step_artifact
            |_timeshare_mem_bound
                |_ fig_8_a.csv
                |_ fig_8_b.csv
                |_ figure12.pdf
        ```

### Run and Validate Figure 13 (5 human-minutes + 50 compute-minutes)

* Run the following commands

    ```bash
    ### In the docker container ###
    cd /root/step_artifact/
    source setup.sh
    pytest timeshare_mem_bound/test_membound_qwen_sweep_revet.py::test_static_tile
    # Produced files: step_artifact/timeshare_mem_bound/fig_9_a.csv,
    #                 step_artifact/timeshare_mem_bound/fig_9_b.csv

    python timeshare_mem_bound/generate_fig13.py 
    # Produced file: step_artifact/timeshare_mem_bound/figure13.pdf

    ```

  * The `test_static_tile` will run experiments for figure 13 and produce `step_artifact/timeshare_mem_bound/fig_9_a.csv` and `step_artifact/timeshare_mem_bound/fig_9_b.csv`.

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 13 as follows:

        ```python
        FILES_TO_COPY = [
            "step_artifact/timeshare_mem_bound/fig_9_a.csv",
            "step_artifact/timeshare_mem_bound/fig_9_b.csv",
            "step_artifact/timeshare_mem_bound/figure13.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `timeshare_mem_bound` folder. The file `figure13.pdf` should match Figure 13 in the paper. The values used to create the plot are in `fig_9_a.csv` and `fig_9_b.csv`.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_step_artifact
            |_timeshare_mem_bound
                |_ fig_9_a.csv
                |_ fig_9_b.csv
                |_ figure13.pdf
        ```

### Run and Validate Figure 14 (5 human-minutes + 4 compute-minutes)

* Run the following commands

    ```bash
    ### In the docker container ###
    cd /root/step_artifact/
    source setup.sh
    pytest dynamic_par/sweep_ae_revision.py::test_b64_sweep # 3m30s
    # Produced file: step_artifact/dynamic_par/batch64_interleave_dynamic.csv

    python dynamic_par/fig_interleave_dyn.py
    # Produced file: step_artifact/dynamic_par/figure14.pdf

    echo "figure 14 done"

    ```

  * The `test_b64_sweep` will run experiments for figure 14 and produce `step_artifact/dynamic_par/batch64_interleave_dynamic.csv`.

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 14 as follows:

        ```python
        FILES_TO_COPY = [
            "step_artifact/dynamic_par/batch64_interleave_dynamic.csv",
            "step_artifact/dynamic_par/figure14.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `dynamic_par` folder. The file `figure14.pdf` should match Figure 14 in the paper. The values used to create the plot are in the other three CSV files.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_step_artifact
            |_dynamic_par
                |_ batch64_interleave_dynamic.csv
                |_ figure14.pdf
        ```

### Run and Validate Figure 15 (5 human-minutes + 1 compute-minutes)

* Run the following commands

    ```bash
    ### In the docker container ###
    cd /root/step_artifact/
    source setup.sh
    pytest dynamic_par/sweep_ae_revision.py::test_batch_sweep # 1m
    # Produced file: step_artifact/dynamic_par/batch_sweep_coarse_vs_dynamic.csv

    python dynamic_par/fig_coarse_dyn_64.py
    # Produced file: step_artifact/dynamic_par/figure15.pdf

    echo "figure 15 done"

    ```

  * The `test_batch_sweep` will run experiments for figure 14 and produce `step_artifact/dynamic_par/batch_sweep_coarse_vs_dynamic.csv`.

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 15 as follows:

        ```python
        FILES_TO_COPY = [
            "step_artifact/dynamic_par/batch_sweep_coarse_vs_dynamic.csv",
            "step_artifact/dynamic_par/figure15.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `dynamic_par` folder. The file `figure15.pdf` should match Figure 15 in the paper. The values used to create the plot are in the other three CSV files.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_step_artifact
            |_dynamic_par
                |_ batch_sweep_coarse_vs_dynamic.csv
                |_ figure15.pdf
        ```

### Run and Validate Figure 21 (5 human-minutes + 15 compute-minutes)

* Run the following commands

    ```bash
    ### In the docker container ###
    cd /root/step_artifact/
    source setup.sh
    pytest dynamic_par/sweep_ae.py::test_b16_sweep
    # Produced file: step_artifact/dynamic_par/batch16_sweep_ae.csv

    pytest dynamic_par/sweep_ae.py::test_b64_sweep
    # Produced file: step_artifact/dynamic_par/batch64_sweep_ae.csv

    pytest dynamic_par/sweep_ae.py::test_b64_b16_sweep
    # Produced file: step_artifact/dynamic_par/batch80_sweep_ae.csv

    python dynamic_par/fig21_change_scale.py
    # Produced file: step_artifact/dynamic_par/figure21.pdf
    echo "figure 21 done"

    ```

  * The `test_b16_sweep` will run experiments for `B = 16` (left) in figure 21 and produce `step_artifact/dynamic_par/batch16_sweep_ae.csv`.
  * The `test_b64_sweep` will run experiments for `B = 64` (middle) in figure 21 and produce `step_artifact/dynamic_par/batch64_sweep_ae.csv`.
  * The `test_b16_sweep` will run experiments for `B = 64+16` (right) in figure 21 and produce `step_artifact/dynamic_par/batch80_sweep_ae.csv`.

* To validate the results:
    1. Exit the docker (CTRL+p, CTRL+q) and move into the cloned `step_artifact` repository on the local machine.

        ```bash
        # Exit the docker (CTRL+p, CTRL+q)
        
        ### In the local machine ###
        $ cd step_artifact
        ```

    2. modify the `FILES_TO_COPY` list in the `step_artifact/copy_from_docker.py` file to only include the files related to figure 21 as follows:

        ```python
        FILES_TO_COPY = [
            "step_artifact/dynamic_par/batch16_sweep_ae.csv",
            "step_artifact/dynamic_par/batch64_sweep_ae.csv",
            "step_artifact/dynamic_par/batch80_sweep_ae.csv",
            "step_artifact/dynamic_par/figure21.pdf",
        ]
        ```

    3. Run the following command. This will copy the experiment results and figures from the container. The results and figures will be copied to `step_artifact/<OUTPUT_DIRECTORY>`.

        ```bash
        ### In the local machine ###
        $ mkdir -p <OUTPUT_DIRECTORY>  # This will be the argument for the --output_dir in the following line
        $ python copy_from_docker.py --docker_id <CONTAINER_ID> --output_dir <OUTPUT_DIRECTORY>
        ```

    4. The reproduced figure and experiment results can be found in the `dynamic_par` folder. The file `figure21.pdf` should match Figure 21 in the paper. The values used to create the plot are in the other three CSV files.

        ```
        step_artifact/<OUTPUT_DIRECTORY>
        |_step_artifact
            |_dynamic_par
                |_ batch16_sweep_ae.csv
                |_ batch64_sweep_ae.csv
                |_ batch80_sweep_ae.csv
                |_ figure21.pdf
        ```

## [Optional] To customise or extend the toolchain

As an abstraction, STeP is not tied to a specific hardware implementation and is portable across diverse Spatial Dataflow Accelerator (SDA) implementations with software-managed scratchpads (similar to [The Sparse Abstract Machine](https://dl.acm.org/doi/10.1145/3582016.3582051).)
We will walk through how the symbolic Python frontend (`src` folder) and the simulator (`step-perf`) can be customized or extended.

### Symbolic frontend

* Changing existing equations for off-chip traffic and on-chip memory requirement:
  * The symbolic frontend implements symbolic expressions for off-chip memory traffic and on-chip memory requirements for each operator using SymPy.
  * The expressions can be customized to capture hardware-specific operator details, such as hardware tile sizes and matrix-multiplication implementation. For example, below is the symbolic expression equation for `LinearOffChipLoad`. While we multiply 2 assuming double buffering, one can change it to multiply only 1 if the target SDA does not support double buffering.

    ```python
    def off_chip_traffic(self) -> sympy.Expr:
        """Return the off-chip traffic for this operation."""
        total_elements = self._stream.total_elements() * sympy.Integer(
            self.tile_row * self.tile_col * self.n_byte
        )
        return total_elements

    def on_chip_requirement(self, count_fifos: bool = False) -> sympy.Expr:
        """Return the on-chip memory requirement for this operation."""
        return sympy.Integer(self.tile_row * self.tile_col * self.n_byte * 2)
    ```

* Equations for other metrics:
  * The frontend includes equations for off-chip traffic and on-chip memory usage as the applications we experimented are (off-chip) memory-bound. However, if performance bottlenecks shift, additional cost functions can be added to STeP operators to obtain performance-correlated metrics (e.g. on-chip traffic, compute). For example, if the bottleneck shifts to the boundary between on-chip memory and PE storage, the programmer can update the base class for the STeP operators to include a function for on-chip memory traffic and implement them for the STeP operators.

    ```python
    class StepOps(ABC):
        _counter: int = 0
        instance_id: int

        ...

        @abstractmethod
        def on_chip_traffic(self, count_fifos: bool = False) -> sympy.Expr:
            """Return the on-chip memory traffic for this operation."""
            pass

        @abstractmethod
        def off_chip_traffic(self) -> sympy.Expr:
            """Return the off-chip traffic (bytes) for this operation."""
            pass
    ```

### Simulator

The simulator builds on top of the [Dataflow Abstract Machine simulation framework (DAM)](https://ieeexplore.ieee.org/document/10609587). Each STeP operator is implemented as a *context* in DAM and FIFOs are implemented using DAM's *channels*.

* Operator initiation intervals and latencies can be adjusted to match hardware characteristics. For example, below is an example context definition for Map:

    ```rust
    #[context_macro]
    pub struct Map<E, T: DAMType, OT: DAMType> {
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<OT>>>,
        func: Arc<dyn Fn(&Tile<T>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, FLOPs per cycle -> cycles
        config: MapConfig,
        id: u32,
        _phantom: PhantomData<E>,
    }

    ```

    The timing behavior of the operator is implemented in the `run` function for each context. As we use Roofline model in our simulator, it calculates the latency based on the input data and the function and increments the time of the node by the latency calculated.

    ```rust
    impl<
            E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
            T: DAMType,
            OT: DAMType,
        > Context for Map<E, T, OT>
    where
        Elem<Tile<T>>: DAMType,
        Elem<Tile<OT>>: DAMType,
    {
        fn run(&mut self) {
            loop {
                let in_elem = self.in_stream.peek_next(&self.time);
                let (in_tile, stop_lev) = match in_elem {
                    ...
                };

                let start_time = self.time.tick().time();
                let load_cycles = if in_tile.read_from_mu {
                    div_ceil(in_tile.size_in_bytes() as u64, PMU_BW)
                } else {
                    0
                };

                let (comp_cycles, out_tile) =
                    (self.func)(&in_tile, self.config.compute_bw, self.config.write_back_mu);
                let store_cycles = if self.config.write_back_mu {
                    div_ceil(out_tile.size_in_bytes() as u64, PMU_BW)
                } else {
                    0
                };

                let roofline_cycles = [load_cycles, comp_cycles, store_cycles]
                    .into_iter()
                    .max()
                    .unwrap_or(0);

                self.time.incr_cycles(roofline_cycles); // <= Latency

                let data = match stop_lev {
                    Some(level) => Elem::ValStop(out_tile, level),
                    None => Elem::Val(out_tile),
                };
                self.out_stream
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: self.time.tick(), // <= time the result appears in the output FIFO
                            data: data,
                        },
                    )
                    .unwrap();

                self.in_stream.dequeue(&self.time).unwrap();
            }
        }
    }
    ```

* Different memory technologies:
  * Different memory technologies can be integrated by building a DAM context that makes library calls to the memory simulator (e.g., Ramulator2).
  * As shown below, STeP's off-chip memory operators includes channels that communicate with the memory simulator. The programmer has to connect a channel pair (`addr_snd`, `resp_addr_rcv`) between the memory simulator and STeP's offchip memory operators.

    ```rust
    #[context_macro]
    pub struct OffChipLoad<E: LoggableEventSimple, T: DAMType> {
        ...
        // Sender & Receiver (DAM details)
        pub addr_snd: Sender<ParAddrs>,   // => to memory simulator
        pub resp_addr_rcv: Receiver<u64>, // <= from memory simulator
        pub on_chip_snd: Sender<Elem<Tile<T>>>, // => on chip memory unit
        pub id: u32,
        _phantom: PhantomData<E>, // Needed to use the generic parameter E
    }
    ```

  * The off-chip memory operator simulates the memory access delays by sending an address to the DAM context for the memory simulator and then sending the data to the next unit once it receives a response from the memory simulator through the  `resp_addr_rcv` channel for that address.
