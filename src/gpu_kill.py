#!/usr/bin/env python3
import subprocess
import signal
import os
import argparse
import sys
import time

def get_gpu_processes():
    """Get all processes using GPUs using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_name', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():  # Skip empty lines
                pid, process_name, gpu_name = [x.strip() for x in line.split(',', 2)]
                processes.append({
                    'pid': int(pid),
                    'name': process_name,
                    'gpu': gpu_name
                })
        return processes
    except subprocess.CalledProcessError:
        print("Error: Could not get GPU processes. Is nvidia-smi installed?")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def get_python_processes():
    """Get specific Python script processes and accelerate launch processes."""
    target_scripts = ['unmemorizerun.py', 'run_experiments.py', 'run_all_experiments.sh', 'test.py', 'unmemorize.sh', 'run_all_experiments.sh']
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True)
        processes = []
        for line in result.stdout.strip().split('\n'):
            # Check for target scripts
            if any(script in line for script in target_scripts):
                parts = line.split()
                try:
                    processes.append({
                        'pid': int(parts[1]),
                        'name': next(script for script in target_scripts if script in line),
                        'gpu': 'N/A'
                    })
                except (IndexError, ValueError):
                    continue
            # Check for accelerate launch processes
            elif 'accelerate' in line and 'launch' in line:
                parts = line.split()
                try:
                    processes.append({
                        'pid': int(parts[1]),
                        'name': 'accelerate-launch',
                        'gpu': 'N/A'
                    })
                except (IndexError, ValueError):
                    continue
        return processes
    except Exception as e:
        print(f"Error getting Python processes: {e}")
        return []
        
def kill_process(pid, force=False):
    """Kill a process by PID."""
    try:
        if force:
            os.kill(pid, signal.SIGKILL)
            return True
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.kill(pid, 9)
                print(f"Process {pid} did not terminate with SIGTERM")
                return False
            except ProcessLookupError:
                return True
    except ProcessLookupError:
        print(f"Process {pid} not found")
        return False
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Kill all processes using GPUs and specific Python scripts')
    parser.add_argument('-f', '--force', action='store_true',
                       help='Force kill processes (SIGKILL instead of SIGTERM)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    # Get all target processes
    while True:
        gpu_processes = get_gpu_processes()
        python_processes = get_python_processes()
        all_processes = gpu_processes + python_processes
        
        if not all_processes:
            print("No target processes found.")
            return
        
        # Print processes that will be killed
        print("The following processes will be terminated:")
        for proc in all_processes:
            print(f"PID: {proc['pid']}, Name: {proc['name']}, GPU: {proc['gpu']}")
        
        # Kill processes
        success_count = 0
        for proc in all_processes:
            pid = proc['pid']
            if args.verbose:
                print(f"\nAttempting to kill process {pid} ({proc['name']})...")
            
            if kill_process(pid, args.force):
                success_count += 1
                if args.verbose:
                    print(f"Successfully terminated process {pid}")
            
        # Print summary
        print(f"\nTerminated {success_count} out of {len(all_processes)} processes.")
        
if __name__ == "__main__":
    main()