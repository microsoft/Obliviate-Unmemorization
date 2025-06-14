import os
import torch
import argparse

lm_eval_tasks = "mmlu,winogrande,truthfulqa,hellaswag"



def run_benchmark( logging_folder, model_folder, overwrite=True ):
    # get the number of GPUs using torch
    num_gpus = torch.cuda.device_count()
    commandLine = f"accelerate launch --config_file ../config/accelerate/accelerate_benchmark_config{num_gpus}.yaml -m lm_eval --trust_remote_code"
    commandLine += f" --tasks {lm_eval_tasks}"
    commandLine += f" --model_args pretrained={model_folder},trust_remote_code=True"

    output_dir = f"{logging_folder}/benchmark_results"
    if os.path.exists(output_dir ):
        if overwrite:
            # delete directory
            os.system(f"rm -rf {output_dir}")
        else:
            return True
    commandLine += f" --output_path={output_dir}"

    result = os.system(commandLine)
    if result != 0:
        return False
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The model folder to be benchmarked",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The folder to store the benchmark results",
    )
    return parser.parse_args()
    

def main():
    args = parse_args()
    run_benchmark(args.output_dir, args.model_name_or_path)

if __name__ == "__main__":
    main()