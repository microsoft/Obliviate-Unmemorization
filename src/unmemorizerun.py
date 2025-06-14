import argparse
import json
import os
import torch

# dictionary of steps to run
run_steps = {
    "unmemorize": True,
    "test": True,
    "benchmark": True,
    "benchmark-pretrained": False,
    "benchmark-memorized": False,
    "test-pretrained": False,
}

lm_eval_tasks = "mmlu,winogrande,truthfulqa,hellaswag"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run SFT fingerprinting.")
    parser.add_argument("--model_name", type=str, required=True, help="Model config name.")
    parser.add_argument("--logging_folder", type=str, required=True, help="logging output folder.")
    parser.add_argument("--model_folder", type=str, default="", help="model folder.")
    parser.add_argument("--dataset", type=str, default="data/synthetic", help="Dataset to use.")
    parser.add_argument("--notrain", action="store_true", help="Do not train the model.")
    parser.add_argument("--fresh", action="store_true", help="Force fresh run even if model exists")   
    
    # stages to run
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the model.")
    parser.add_argument("--benchmark-pretrained", action="store_true", help="Benchmark the pretrained model.")
    parser.add_argument("--benchmark-memorized", action="store_true", help="Benchmark the memorized model.")
    parser.add_argument("--test-pretrained", action="store_true", help="Test the pretrained model.")

    # run configuration
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs.")
    parser.add_argument("--run_config", type=str, help="Run config file.")    
    parser.add_argument("--smart_stride", action="store_true", help="Find good token stride.")
    parser.add_argument("--smart_select", action="store_true", help="Find good tokens.")
    parser.add_argument("--unmemorize_sample_count", type=int, default=-1, help="Number of samples to memorize.")
    parser.add_argument("--top_k", type=int, default=10, help="Top k tokens for k/l loss.")
    parser.add_argument("--loss_type", type=str, default=None, choices=[None, "kl", "combined"], help="Loss type to use for unmemorization.")

    # multi GPU node configs
    parser.add_argument("--accelerate_run_config", type=str, default=None, help="Accelerate config file.")
    parser.add_argument("--accelerate_benchmark_config", type=str, default=None, help="Accelerate config path.")

    args = parser.parse_args()

    # if any steps are explicitly listed, only those steps are run. First check if any 
    # step is explicitly listed, then update the run_steps dictionary
    if args.test or args.benchmark or args.benchmark_pretrained or args.benchmark_memorized or args.test_pretrained:
        run_steps["unmemorize"] = False
        run_steps["test"] = args.test
        run_steps["benchmark"] = args.benchmark
        run_steps["benchmark-pretrained"] = args.benchmark_pretrained
        run_steps["benchmark-memorized"] = args.benchmark_memorized
        run_steps["test-pretrained"] = args.test_pretrained
        
    if args.notrain:
        run_steps["unmemorize"] = False

    print("unmemorizerun.py arguments:")
    print('\n   '.join(['{}: {}'.format(k, v) for k, v in vars(args).items()]))
    print("")
    return args

def load_model_config(model_name):
    # load the model config file under config/model_config.json
    model_config_path = os.path.join("config", "model_config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    
    # model name is after the last '/' if there is one
    model_name = model_name.split("/")[-1]
    if model_name not in model_config:
        raise ValueError("Model name not found in model_config.json")
    return model_config[model_name]

def should_run_unmemorize(model_dir: str, force_fresh: bool) -> bool:
    """Check if unmemorize should run based on existing config and fresh flag"""
    config_path = os.path.join(model_dir, "config.json")
    if force_fresh:
        return True
    return not os.path.exists(config_path)

def run_memorize( model_name, model_config, logging_folder, model_folder, dataset, sample_count, accelerate_config_path=None ):
    
    # if there's a config.json file in the model folder, then we don't need to memorize
    if os.path.exists(os.path.join(model_folder, "config.json")):
        print("Model already exists. Skipping memorize.")
        return True
        
    # delete the log
    logfile = logging_folder + "/memorize.log"
    if os.path.exists(logfile):
        os.remove(logfile)
        
    commandLine = "accelerate launch"
    
    if accelerate_config_path is not None:
        commandLine += f" --config_file {accelerate_config_path}"
    
    commandLine += " src/finetune.py"
    
    # add the model name
    commandLine += " --model_name " + model_name
    if model_config.get("instruct", None) is not None:
        commandLine += " --instruct" 
    commandLine += " --logfile " + logfile
    commandLine += " --output_dir " + model_folder
        
    # training parameters
    per_device_batch_size = model_config.get("per_device_batch_size", 4)
    commandLine += f" --per_device_train_batch_size {per_device_batch_size}"
    gradient_accumulation_steps = model_config.get("gradient_accumulation_steps", 2)
    commandLine += f" --gradient_accumulation_steps {gradient_accumulation_steps}"
    
    # memorize parameters
    commandLine += f" --unmemorize_sample_count {sample_count}"
    commandLine += f" --dataset_name {dataset}" 
    
    # run the command line and return False if it fails
    result = os.system(commandLine)
    if result != 0:
        return False
    return True


def run_unmemorize(model_name, model_config, logging_folder, model_folder,
                   dataset, start, stride, span, smart_stride, smart_select, sample_count,
                   top_k, loss_type=None, accelerate_config_path=None):
    
    # delete the log
    logfile = logging_folder + "/unmemorize.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    
    commandLine = "accelerate launch"

    if accelerate_config_path is not None:
        commandLine += f" --config_file {accelerate_config_path}"

    commandLine += " src/finetune.py"
    
    # add the model name
    commandLine += " --model_name " + model_name
    if model_config.get("instruct", None) is not None:
        commandLine += " --instruct" 
    if model_config.get("tokenizer", None) is not None:
        commandLine += " --tokenizer_name " + model_config["tokenizer"]
    commandLine += " --logfile " + logfile
    commandLine += " --output_dir " + model_folder
        
    # training parameters
    if model_config.get("learning_rate", None) is not None:
        commandLine += f" --learning_rate {model_config['learning_rate']}"
    else:
        commandLine += " --learning_rate 1e-6"
    per_device_batch_size = model_config.get("per_device_batch_size", 4)
    commandLine += f" --per_device_train_batch_size {per_device_batch_size}"
    gradient_accumulation_steps = model_config.get("gradient_accumulation_steps", 1)
    commandLine += f" --gradient_accumulation_steps {gradient_accumulation_steps}"
    
    # unmemorize parameters
    commandLine += f" --unmemorize"
    if smart_stride:
        commandLine += f" --unmemorize_smart_stride"
    if smart_select:
        commandLine += f" --unmemorize_smart_select"        
    commandLine += f" --dataset_name {dataset}" 
    commandLine += f" --unmemorize_start {start}"
    commandLine += f" --unmemorize_stride {stride}"
    commandLine += f" --unmemorize_span {span}"
    commandLine += f" --unmemorize_sample_count {sample_count}"
    commandLine += f" --unmemorize_top_k {top_k}"
    
    # add loss_type parameter if provided
    if loss_type is not None:
        commandLine += f" --loss_type {loss_type}"
    
    # run the command line and return False if it fails
    result = os.system(commandLine)
    if result != 0:
        return False
    return True
    
    
def run_test( model_name, logging_folder, dataset, sample_count, instruct, runMIA=False ):

    # if test.log doesn't exists, execute the tests 
    if os.path.exists(logging_folder + "/test.json"):
        print("Test already exists. Skipping test.")
    else:    

        # run rest of tests
        commandLine = "python src/test.py"
        commandLine += " --model " + model_name
        commandLine += " --dataset " + dataset
        commandLine += " --logging_folder " + logging_folder 
        commandLine += f" --sample_count {sample_count}"    
        if instruct is True:
            commandLine += " --instruct"

        result = os.system(commandLine)
        if result != 0:
            return False  

    if os.path.exists(logging_folder + "/test-greedy.json"):
        print("Test greedy already exists. Skipping test.")
    else:
        # run rest of tests
        commandLine = "python src/test.py"
        commandLine += " --model " + model_name
        commandLine += " --dataset " + dataset
        commandLine += " --logging_folder " + logging_folder 
        commandLine += f" --sample_count {sample_count}"    
        if instruct is True:
            commandLine += " --instruct"
                    
        # now run with greedy selection
        commandLine += " --greedy"
        
        result = os.system(commandLine)
        if result != 0:
            return False
    return True

def run_benchmark( model_name, logging_folder, accelerate_config_path=None ):
    
    # if benchmark.log exists, then we don't need to run the benchmark
    if os.path.exists(logging_folder + "/benchmark.log"):
        print("Benchmark already exists. Skipping benchmark.")
        return True
    
    # get the number of GPUs using torch
    num_gpus = torch.cuda.device_count()
    if accelerate_config_path is not None:
        commandLine = f"accelerate launch --config_file {accelerate_config_path}"
    else: 
        commandLine = f"accelerate launch --config_file config/accelerate/accelerate_benchmark_config{num_gpus}.yaml"
    commandLine += " -m lm_eval --trust_remote_code"
    commandLine += f" --tasks {lm_eval_tasks}"
    commandLine += f" --model_args pretrained={model_name},trust_remote_code=True"

    output_path = f"{logging_folder}"
    commandLine += f" > {output_path}/benchmark.log"

    result = os.system(commandLine)
    if result != 0:
        return False
    return True


def main():
    # get the model name
    args = parse_arguments()
    
    model_config = load_model_config(args.model_name)
    instruct_model = model_config.get("instruct", False)    
    print(model_config)        
    
    # do we need to benchmark the pretrained?
    if run_steps["benchmark-pretrained"] or run_steps["benchmark-memorized"] or run_steps["test-pretrained"]:
        if run_steps["benchmark-pretrained"] or run_steps["test-pretrained"]:
            run_logging_folder = args.logging_folder 
            model_name = model_config["model_name"]
        else:
            run_logging_folder = args.logging_folder + "/memorized"
            base_model_name = model_config["model_name"]
            model_folder = run_logging_folder + "/model"
            model_name = model_folder

            # memorize
            if run_memorize( base_model_name, model_config, run_logging_folder, model_folder,
                             args.dataset,
                             args.unmemorize_sample_count,
                             args.accelerate_run_config ) == False:
                print(f"Error in memorize step")
                return
            
        if run_test(model_name, run_logging_folder, args.dataset,
                    args.unmemorize_sample_count, instruct_model ) == False:
            print(f"Error in test step")
            return

        if run_steps["benchmark-pretrained"] is False:
            if run_benchmark( model_name, run_logging_folder, args.accelerate_benchmark_config ) == False:
                print(f"Error in benchmark step")
                return
        return
    else:
        model_name = args.model_folder
    
    # load the json run config file
    with open(args.run_config, "r") as f:
        run_config = json.load(f)

    # extract the base of hte run_config file name
    run_config_base = os.path.basename(args.run_config).split(".")[0]
    
    logging_folder = args.logging_folder + "/" + run_config_base
    os.makedirs(logging_folder, exist_ok=True)
    
    # create folder for model and fingerprints
    if args.model_folder == "":
        args.model_folder = logging_folder   

    # run the number of runs        
    for run_number in range(args.num_runs):

        run_logging_folder = logging_folder + "/" + str(run_number)

        # does the model already exists?
        run_model_folder = logging_folder + "/" + str(run_number)
        if args.notrain == False:
                    
            print(f"\n\n[{run_number}] ############# Starting run")
            
            # create the output folders
            os.makedirs(run_logging_folder, exist_ok=True)
            os.makedirs(run_model_folder, exist_ok=True)        
                
            # run the fingerprinting step
            if run_steps["unmemorize"]:
                if should_run_unmemorize(run_model_folder, args.fresh):
                    print(f"\n[{run_number}] ************* Unmemorizing")
                    if run_unmemorize(model_name, model_config, run_logging_folder, run_model_folder,
                                    args.dataset, run_config["start"], run_config["stride"], run_config["span"],
                                    args.smart_stride, args.smart_select,
                                    args.unmemorize_sample_count,
                                    args.top_k,
                                    args.loss_type,
                                    args.accelerate_run_config ) == False:
                        print(f"[{run_number}] Error in unmemorize")        
                        break
                else:
                    print(f"\n[{run_number}] ************* Skipping unmemorize - model already exists (use --fresh to force rerun)")
                
                # run the test step
                if run_steps["test"]:
                    print(f"\n[{run_number}] ************* Running tests")
                    if run_test(run_model_folder, run_logging_folder, args.dataset, 
                                args.unmemorize_sample_count, instruct_model) == False:
                        print(f"[{run_number}] Error in test step")
                        break
                        
                # run the benchmark step
                if run_steps["benchmark"]:
                    print(f"\n[{run_number}] ************* Running benchmarks")
                    if run_benchmark( run_model_folder, run_logging_folder, args.accelerate_benchmark_config ) == False:
                        print(f"[{run_number}] Error in benchmark step")
                        break                                
                            
        
if __name__ == "__main__":
    main()



