import torch
import numpy as np

import json

import re
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from networkx import DiGraph
from numpy.typing import NDArray


def reconstruct_numpy(output_path: str, delete_npy: bool = False) -> NDArray:
    with open(f"{output_path}.json", "r") as f:
        vec = json.load(f)

    array: NDArray = np.load(f"{output_path}.npy")

    if delete_npy is True:
        # Specify the file path
        data_file_path = f"{output_path}.npy"
        meta_file_path = f"{output_path}.json"

        # Check if the file exists before deleting
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
            print(f"{data_file_path} has been deleted after data retrieval")
        else:
            print(f"{data_file_path} does not exist.")

        if os.path.exists(meta_file_path):
            os.remove(meta_file_path)
            print(f"{meta_file_path} has been deleted after data retrieval")
        else:
            print(f"{meta_file_path} does not exist.")

    return array.reshape(vec)


def check_gold_npy(sim_out_path, gold_path):
    out_sim = reconstruct_numpy(sim_out_path, delete_npy=False)
    gold: NDArray = np.load(f"{gold_path}.npy")
    try:
        torch.testing.assert_close(
            torch.from_numpy(gold), torch.from_numpy(out_sim), rtol=1e-4, atol=1e-6
        )
        print("Congratulations! Test passed!")
    except AssertionError as e:
        print("Mismatch found:", e)


def check_gold_tensor(sim_out_path, gold: torch.Tensor) -> bool:
    out_sim = reconstruct_numpy(sim_out_path, delete_npy=False)
    print("Output shape:", out_sim.shape)
    print("Gold shape:", gold.shape)
    try:
        torch.testing.assert_close(
            gold, torch.from_numpy(out_sim), rtol=1e-4, atol=1e-6
        )
        print("Congratulations! Test passed!")
        return True
    except AssertionError as e:
        print("Mismatch found:", e)
        print("Output tensor:", out_sim)
        print("Gold tensor:", gold.numpy())
        return False
