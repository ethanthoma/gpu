import os
import shutil
import subprocess
import sys
import warnings

import torch
import triton.backends.nvidia.driver as triton_driver
from reference import check_implementation, generate_input
from submission_tinygrad import custom_kernel

warnings.filterwarnings("ignore", category=UserWarning, module="torch.backends")

ldconfig_path = shutil.which("ldconfig")
nvidia_lib = "/nix/store/a70fzhggs1728k80pn1rvzjw0rbjanvv-nvidia-x11-580.95.05-6.12.57/lib"

original_libcuda_dirs = triton_driver.libcuda_dirs


def patched_libcuda_dirs():
    try:
        libs = subprocess.check_output([ldconfig_path, "-p"], stderr=subprocess.DEVNULL).decode(errors="ignore")
        return [line.split()[-1] for line in libs.splitlines() if "libcuda.so" in line]
    except:
        return []


triton_driver.libcuda_dirs = patched_libcuda_dirs

original_library_dirs = triton_driver.library_dirs


def patched_library_dirs():
    dirs = original_library_dirs()
    if nvidia_lib not in dirs:
        dirs.append(nvidia_lib)
    return dirs


triton_driver.library_dirs = patched_library_dirs


def test_all_cases():
    test_cases = [
        {"size": 1023, "seed": 4242},
        {"size": 1024, "seed": 5236},
        {"size": 1025, "seed": 1001},
        {"size": 2048, "seed": 5531},
        {"size": 4096, "seed": 9173},
        {"size": 2**22, "seed": 1739},
    ]

    print("Running local tests...")
    all_passed = True

    for i, test in enumerate(test_cases):
        data = generate_input(**test)
        data_copy = (data[0].clone(), data[1].clone())

        output = custom_kernel(data)
        passed, message = check_implementation(data_copy, output)

        status = "" if passed else ""
        print(f"{status} Test {i + 1}: size={test['size']}, seed={test['seed']}")
        if not passed:
            print(f"  Error: {message}")
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")

    return all_passed


if __name__ == "__main__":
    test_all_cases()
