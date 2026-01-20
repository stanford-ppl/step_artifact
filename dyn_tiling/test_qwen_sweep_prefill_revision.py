from dataclasses import dataclass
import numpy as np
import torch
import csv
import time
from datetime import datetime, timezone

from dyn_tiling.test_weight_stationary_gemm import run_ws_tile_mn_mk
from dyn_tiling.test_weight_stationary_gemm_dyn_tile import run_ws_tile_mn_mk_dyn_tile
from dyn_tiling.test_weight_stationary_gemm_revet import run_ws_tile_mn_mk_revet

# from step_py.ops import *
# from step_py.functions import map_accum_fn, map_fn, init_fn, accum_fn
# from utils.gold_checking import check_gold_tensor
# from utils.draw_graph import save_graph_format
# from rewrite.broadcast import infer_broadcast
# from utils.moe import *


@dataclass
class SmallerQwen30b:  # 16x scaled down version for each dimension
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 128  # 2048 // 16
    moe_inter_dim = 48  # 768 // 16


@dataclass
class Qwen30b:
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 2048  # https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json#L12
    moe_inter_dim = (
        768  # https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json#L19
    )


@dataclass
class TinyQwen30b:  # 32x scaled down version for each dimension
    n_routed_experts = 128
    n_activated_experts = 8
    dim = 64  # 2048 // 32
    moe_inter_dim = 24  # 768 // 32



def test_qwen_b1024(i_id_arg=22, l_id_arg=16):
    mock_bf16 = True
    
    chan_depth = None

    # ------------ Model Configuration ------------
    # model_config = SmallerQwen30b()
    model_config = Qwen30b()


    # ------------ Expert Indices ------------
    batch = 1024  # 256, 512, 1024
    # i_id = 22
    # l_id =16
    # i_id = 2
    # l_id =2 
    i_id = i_id_arg
    l_id = l_id_arg   
    # https://github.com/gina7484/expert_routing_data/blob/main/processed_qwen3-30B-A3B_3060_5764/per_iter_per_layer/bincount_analysis.csv
    # i_2_l_2: High (stdev: 241, bincount max: 1024, num non-selected experts: 106)
    # i_22_l_16: Med (stdev:131, bincount max: 814, num non-selected experts: 45)
    # i_22_l_0: Low (stdev:70, bincount max: 324, num non-selected experts: 19)
    expert_selection_file = f"./dyn_tiling/expert_routing/qwen_b{batch}/{i_id:03d}_{l_id:03d}.npz"
    expert_indices_npz = np.load(expert_selection_file)
    expert_indices = torch.from_numpy(
        expert_indices_npz["data"]
    )  # [B, n_activated_experts]

    # expert_counts: [n_routed_experts] (bincount across all batches)
    expert_counts = torch.bincount(
        expert_indices.flatten(), minlength=model_config.n_routed_experts
    )
    print(f"Expert counts: {expert_counts}")
    # [  0,   1,   0,   0,  11,   0,   5, 326,  88, 201,   0, 304,  34,  25,
        #  50, 335,  45, 363, 617,   0, 203,   0,   1,   1,  42,   2,  12,   4,
        #   1,  76, 101,   0,   0,  41,   0,   2,   2,  32,   0,  30,  74,   1,
        #   3, 204,  22, 697,   0,   1,   0,   0,  37,  42,   9,   4,   0, 175,
        #   0,  66,   9,   0,  40,   0,   1,   0,   0, 100,   1,   0,   0,   0,
        #   1,   0,  17, 406,   0, 201,   6, 116, 275,  94,  73,   1,   1, 814,
        # 107,   0,   0,   0,   0,   6,   1,   5,   0,   1,  20,   0,   0,   0,
        #   7,   0,   4,   4, 403,  29,   0,   4,   0,   0,  60,  10,  22,   5,
        #   0,   0,  30, 228,   0, 231,   0,  95,   0,  77,   0, 188, 147,   0,
        #   0,  62]

    # ------------ Input generation -----------
    tile_F = 64 if isinstance(model_config, Qwen30b) else 24

    B = expert_indices.shape[0]

    # Set the random seed
    seed = 5
    torch.manual_seed(seed)

    input_tensor = torch.randn(B, model_config.dim)

    # ------------ Result Dict ------------
    result_dict = {
        "cycles": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
        "off_chip_traffic": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
        "on_chip_mem": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
        "on_chip_mem_new": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
    }

    result_dict_raw = {
        "cycles": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
        "off_chip_traffic": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
        "on_chip_mem": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
        "on_chip_mem_new": {
            "tile=16": 0,
            "tile=32": 0,
            "tile=64": 0,
            "tile=128": 0,
            "tile=256": 0,
            "tile=512": 0,
            "tile=1024": 0,
            "tile=dynamic": 0,
        },
    }

    ################## Dynamic Tile Size ##################
    # Start timing
    dyn_tile_start_time = time.perf_counter()
    now = datetime.now(timezone.utc)
    print(f"Start dynamic tile: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    round_N = 1

    dyn_cycles = 1
    dyn_off_chip_traffic = 1
    dyn_on_chip_mem = 1

    (
        off_chip_traffic,
        on_chip_requirement,
        cycles,
        duration_s,
        unit_expert_on_chip,
    ) = run_ws_tile_mn_mk_dyn_tile(
        round_N,
        tile_F,
        input_tensor,
        expert_indices,
        model_config,
        "timing",  # "full",
        False,
        mock_bf16,
        logging=None,
        chan_depth=chan_depth,
    )

    # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
    num_tiles = [
        (routed_toks + round_N - 1) // round_N
        for routed_toks in expert_counts.tolist()
    ]
    after_pad_batch_dim = [num_tiles_i * round_N for num_tiles_i in num_tiles]

    padded_rows = [
        total_toks - raw_toks
        for total_toks, raw_toks in zip(after_pad_batch_dim, expert_counts.tolist())
    ]

    flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in after_pad_batch_dim
        ]
    )

    padded_flops = sum(
        [
            (
                2 * b * model_config.dim * model_config.moe_inter_dim * 3
            )  # 3 (Linear layers)
            + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
            + (
                8 * b * model_config.dim * model_config.moe_inter_dim
            )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
            for b in padded_rows
        ]
    )
    # --------------- off-chip traffic ---------------
    free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

    sub_dict = {
        symbol: value for symbol, value in zip(free_symbols, expert_counts.tolist())
    }
    # print(f"off_chip_traffic: {off_chip_traffic}")

    off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

    # --------------- On-chip requirement ---------------
    free_symbols = sorted(on_chip_requirement.free_symbols, key=str)

    sub_dict = {
        symbol: value for symbol, value in zip(free_symbols, expert_counts.tolist())
    }

    on_chip_requirement_val = on_chip_requirement.subs(sub_dict)

    print(f"on_chip_requirement_val: {on_chip_requirement_val}")
    on_chip_requirement_val = (
        on_chip_requirement_val
        - expert_counts.tolist().count(0) * unit_expert_on_chip
    )


    dyn_cycles = int(cycles)
    dyn_off_chip_traffic = int(off_chip_traffic_val)
    dyn_on_chip_mem = int(on_chip_requirement_val)

    result_dict_raw["cycles"]["tile=dynamic"] = dyn_cycles
    result_dict_raw["off_chip_traffic"]["tile=dynamic"] = dyn_off_chip_traffic
    result_dict_raw["on_chip_mem"]["tile=dynamic"] = dyn_on_chip_mem
    result_dict_raw["on_chip_mem_new"]["tile=dynamic"] = dyn_on_chip_mem

    result_dict["cycles"]["tile=dynamic"] = dyn_cycles / dyn_cycles
    result_dict["off_chip_traffic"]["tile=dynamic"] = dyn_off_chip_traffic / dyn_off_chip_traffic
    result_dict["on_chip_mem"]["tile=dynamic"] = dyn_on_chip_mem / dyn_on_chip_mem
    result_dict["on_chip_mem_new"]["tile=dynamic"] = dyn_on_chip_mem / dyn_on_chip_mem

    # End timing and print duration
    dyn_tile_end_time = time.perf_counter()
    dyn_tile_duration_seconds = int(dyn_tile_end_time - dyn_tile_start_time)
    dyn_tile_duration_minutes = round(dyn_tile_duration_seconds / 60, 2)

    # save results to csv
    out_file = f"./dyn_tiling/figure_10_qwen_b{batch}_raw_dynamic.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "tile_N",
                "cycles",
                "off_chip_traffic",
                "on_chip_mem",
                "on_chip_mem_new",
                "duration_minutes",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            expr = "tile=dynamic"
            writer.writerow(
                    {
                        "tile_N": expr,
                        "cycles": result_dict_raw["cycles"][expr],
                        "off_chip_traffic": result_dict_raw["off_chip_traffic"][expr],
                        "on_chip_mem": result_dict_raw["on_chip_mem"][expr],
                        "on_chip_mem_new": result_dict_raw["on_chip_mem_new"][expr],
                        "duration_minutes": dyn_tile_duration_minutes,
                    }
                )    

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

    # ################## Static Tile Size (256) ##################
    for tile_N in [1024,512,256,128,64,32,16]:
    # for tile_N in [1024,512,256,128]:

        static_i_start_time = time.perf_counter()
        now = datetime.now(timezone.utc)
        print(f"Start static tile {tile_N}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        (
            off_chip_traffic,
            on_chip_requirement,
            cycles,
            duration_s,
            unit_expert_on_chip,
        ) = run_ws_tile_mn_mk_revet(
            tile_N,
            tile_F,
            input_tensor,
            expert_indices,
            model_config,
            "timing",
            False,
            mock_bf16,
            logging=None,
            chan_depth=chan_depth,
            # logging=f"expert_par_gemm_n{tile_N}_f{tile_F}",
        )

        # ------------ substitue symbols in the off_chip_traffic and on_chip_requirement ------------
        num_tiles = [
            (routed_toks + tile_N - 1) // tile_N
            for routed_toks in expert_counts.tolist()
        ]
        after_pad_batch_dim = [num_tiles_i * tile_N for num_tiles_i in num_tiles]

        padded_rows = [
            total_toks - raw_toks
            for total_toks, raw_toks in zip(
                after_pad_batch_dim, expert_counts.tolist()
            )
        ]

        flops = sum(
            [
                (
                    2 * b * model_config.dim * model_config.moe_inter_dim * 3
                )  # 3 (Linear layers)
                + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
                + (
                    8 * b * model_config.dim * model_config.moe_inter_dim
                )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                for b in after_pad_batch_dim
            ]
        )

        padded_flops = sum(
            [
                (
                    2 * b * model_config.dim * model_config.moe_inter_dim * 3
                )  # 3 (Linear layers)
                + b * model_config.moe_inter_dim  # 1 (Element-wise mult)
                + (
                    8 * b * model_config.dim * model_config.moe_inter_dim
                )  # silu_flops: 1(neg)+4(exp)+1(add)+1(div)+1(mul)= 8 FLOPs per element
                for b in padded_rows
            ]
        )

        # --------------- off-chip traffic ---------------
        free_symbols = sorted(off_chip_traffic.free_symbols, key=str)

        sub_dict = {
            symbol: value
            for symbol, value in zip(free_symbols, expert_counts.tolist())
        }

        off_chip_traffic_val = off_chip_traffic.subs(sub_dict)

        # --------------- On-chip requirement ---------------
        on_chip_requirement_val = (
            on_chip_requirement
            - expert_counts.tolist().count(0) * unit_expert_on_chip
        )
        # subtract the on-chip requirement for unselected experts

        cycles = int(cycles)
        off_chip_traffic_val = int(off_chip_traffic_val)
        on_chip_requirement_val = int(on_chip_requirement_val)
        on_chip_requirement_new = int(on_chip_requirement)

        result_dict_raw["cycles"][f"tile={tile_N}"] = cycles
        result_dict_raw["off_chip_traffic"][f"tile={tile_N}"] = off_chip_traffic_val
        result_dict_raw["on_chip_mem"][f"tile={tile_N}"] = on_chip_requirement_val
        result_dict_raw["on_chip_mem_new"][f"tile={tile_N}"] = on_chip_requirement_new

        result_dict["cycles"][f"tile={tile_N}"] = cycles / dyn_cycles
        result_dict["off_chip_traffic"][f"tile={tile_N}"] = off_chip_traffic_val / dyn_off_chip_traffic
        result_dict["on_chip_mem"][f"tile={tile_N}"] = on_chip_requirement_val / dyn_on_chip_mem
        result_dict["on_chip_mem_new"][f"tile={tile_N}"] = on_chip_requirement_new / dyn_on_chip_mem

        static_i_end_time = time.perf_counter()
        static_i_duration_seconds = int(static_i_end_time - static_i_start_time)
        static_i_duration_minutes = round(static_i_duration_seconds / 60, 2)

        # save results to csv
        out_file = f"./dyn_tiling/figure_10_qwen_b{batch}_raw_tile{tile_N}.csv"
        try:
            with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "tile_N",
                    "cycles",
                    "off_chip_traffic",
                    "on_chip_mem",
                    "on_chip_mem_new",
                    "duration_minutes",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

                # Write data rows
                expr = f"tile={tile_N}"
                writer.writerow(
                        {
                            "tile_N": expr,
                            "cycles": result_dict_raw["cycles"][expr],
                            "off_chip_traffic": result_dict_raw["off_chip_traffic"][expr],
                            "on_chip_mem": result_dict_raw["on_chip_mem"][expr],
                            "on_chip_mem_new": result_dict_raw["on_chip_mem_new"][expr],
                            "duration_minutes": static_i_duration_minutes,
                        }
                    )    

            print(f"Results written to {out_file}")
        except Exception as e:
            print(f"Error writing CSV file: {e}")
    ################## Save Results to CSV ##################

    out_file = f"./dyn_tiling/figure_10_qwen_b{batch}.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "metric",
                "tile=16",
                "tile=32",
                "tile=64",
                "tile=128",
                "tile=256",
                "tile=512",
                "tile=1024",
                "tile=dynamic",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for metric, values in result_dict.items():
                writer.writerow(
                    {
                        "metric": metric,
                        "tile=16": values["tile=16"],
                        "tile=32": values["tile=32"],
                        "tile=64": values["tile=64"],
                        "tile=128": values["tile=128"],
                        "tile=256": values["tile=256"],
                        "tile=512": values["tile=512"],
                        "tile=1024": values["tile=1024"],
                        "tile=dynamic": values["tile=dynamic"],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        
    # save results to csv
    out_file = f"./dyn_tiling/figure_10_qwen_b{batch}_raw.csv"
    try:
        with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "tile_N",
                "cycles",
                "off_chip_traffic",
                "on_chip_mem",
                "on_chip_mem_new",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for exper in ["tile=16", "tile=32", "tile=64", "tile=128", "tile=256", "tile=512", "tile=1024", "tile=dynamic"]:
                writer.writerow(
                    {
                        "tile_N": exper,
                        "cycles": result_dict_raw["cycles"][exper],
                        "off_chip_traffic": result_dict_raw["off_chip_traffic"][exper],
                        "on_chip_mem": result_dict_raw["on_chip_mem"][exper],
                        "on_chip_mem_new": result_dict_raw["on_chip_mem_new"][exper],
                    }
                )

        print(f"Results written to {out_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
    
    
def test_qwen_b1024_ablation():
    test_qwen_b1024()