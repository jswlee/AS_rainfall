"""
Progress Tracking Utilities

This module provides functions for tracking progress of the rainfall prediction pipeline
when running from a Jupyter notebook or other Python scripts.
"""

import os
import sys
import subprocess
import time
import threading
import yaml

def run_pipeline_with_progress(project_root, config_path, output_dir=None, estimated_time_seconds=30):
    """
    Run the rainfall prediction pipeline with a progress bar.
    
    Parameters:
    -----------
    project_root : str
        Path to the project root directory
    config_path : str
        Path to the YAML configuration file
    output_dir : str, optional
        Output directory for the pipeline results. If None, uses the output dir from config.
    estimated_time_seconds : int, optional
        Estimated time for the pipeline to complete, used for the progress bar
        
    Returns:
    --------
    dict
        Information about the pipeline run, including success status and timing
    """
    try:
        # Import tqdm here to avoid dependency issues if not in a notebook
        from tqdm.notebook import tqdm
    except ImportError:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Warning: tqdm not installed. Progress bar will not be displayed.")
            # Define a simple fallback if tqdm is not available
            class SimpleTqdm:
                def __init__(self, total, desc):
                    self.total = total
                    self.n = 0
                    self.desc = desc
                    print(f"{desc}: 0/{total}")
                
                def update(self, n):
                    self.n += n
                    print(f"{self.desc}: {self.n}/{self.total}")
                
                def set_description(self, desc):
                    self.desc = desc
                    print(f"Progress: {desc}")
                
                def set_postfix_str(self, postfix):
                    pass
                
                def close(self):
                    print(f"{self.desc}: {self.total}/{self.total} (Complete)")
            
            tqdm = SimpleTqdm
    
    # Get the paths
    pipeline_script = os.path.join(project_root, '2_Create_ML_Data', 'scripts', 'rainfall_prediction_pipeline.py')
    
    if output_dir is None:
        # Load config to get output dir
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Handle different config formats
            if 'paths' in config_data and 'output' in config_data['paths']:
                output_dir = os.path.join(project_root, config_data['paths']['output'])
            elif 'output_dir' in config_data:
                output_dir = config_data['output_dir']
                if not os.path.isabs(output_dir):
                    output_dir = os.path.join(project_root, output_dir)
            else:
                output_dir = os.path.join(project_root, '2_Create_ML_Data', 'output', 'notebook_run')
        except Exception as e:
            print(f"Error loading config file: {e}")
            output_dir = os.path.join(project_root, '2_Create_ML_Data', 'output', 'notebook_run')
    
    # Ensure output_dir is absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    progress_file = os.path.join(output_dir, 'progress.log')
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove any existing progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # Create the command
    cmd = f"python3 {pipeline_script} --config {config_path} --output-dir {output_dir}"
    
    print(f"Running pipeline: {cmd}")
    start_time = time.time()
    
    # Start the process in the background
    process = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Initialize progress tracking
    total_steps = 6  # We know there are 6 steps
    
    # Function to read and display output
    def display_output():
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
    
    # Start output display in a separate thread
    output_thread = threading.Thread(target=display_output)
    output_thread.daemon = True
    output_thread.start()
    
    # Monitor the progress file
    pbar = tqdm(total=total_steps, desc="Pipeline Progress")
    current_step = 0
    step_times = {}
    last_step_time = time.time()
    
    try:
        while process.poll() is None:
            # Check if progress file exists
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    lines = f.readlines()
                    
                # Count progress lines
                progress_lines = [line for line in lines if "PROGRESS:" in line]
                new_step = len(progress_lines)
                
                # Update progress if needed
                if new_step > current_step:
                    # Calculate time for the previous step
                    now = time.time()
                    if current_step > 0:
                        step_times[current_step] = now - last_step_time
                    last_step_time = now
                    
                    # Get the latest progress message
                    if progress_lines:
                        latest = progress_lines[-1].strip()
                        if " - " in latest:
                            step_description = latest.split(" - ")[1]
                            pbar.set_description(f"Step {new_step}/{total_steps}: {step_description}")
                    
                    # Update the progress bar
                    pbar.update(new_step - current_step)
                    current_step = new_step
            
            # Small delay to prevent CPU overuse
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        process.terminate()
        return {"success": False, "interrupted": True}
    finally:
        # Make sure progress bar is completed
        if current_step < total_steps:
            pbar.update(total_steps - current_step)
        
        pbar.close()
    
    # Wait for process to complete
    return_code = process.wait()
    end_time = time.time()
    
    # Check if successful
    success = return_code == 0
    if success:
        print("\nPipeline completed successfully!")
    else:
        print(f"\nPipeline failed with return code {return_code}")
    
    # Read the progress summary
    progress_summary = ""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_summary = f.read()
        print("\nProgress Summary:")
        print(progress_summary)
    
    # Return information about the run
    return {
        "success": success,
        "return_code": return_code,
        "total_time": end_time - start_time,
        "step_times": step_times,
        "progress_summary": progress_summary,
        "output_dir": output_dir
    }

def display_pipeline_timing(result):
    """
    Display timing information from a pipeline run.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from run_pipeline_with_progress
    """
    if not result.get("success", False):
        print("Pipeline did not complete successfully, timing information may be incomplete.")
    
    print(f"\nTotal execution time: {result.get('total_time', 0):.2f} seconds")
    
    step_times = result.get("step_times", {})
    if step_times:
        print("\nTime per step:")
        for step, time_taken in step_times.items():
            print(f"  Step {step}: {time_taken:.2f} seconds")
