"""
UNet Extractor for Stable Diffusion 1.5, SDXL, and FLUX models

This script processes SafeTensors files to extract UNet components.

For enhanced system resource reporting, it's recommended to install psutil:
    pip install psutil

If psutil is not installed, the script will still work but with limited
resource reporting capabilities.
"""

import argparse
import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import threading
import queue
import multiprocessing
import time
import os
import traceback

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def check_resources():
    cpu_count = os.cpu_count() or 1
    
    if PSUTIL_AVAILABLE:
        total_ram = psutil.virtual_memory().total / (1024 ** 3)  # in GB
        available_ram = psutil.virtual_memory().available / (1024 ** 3)  # in GB
    else:
        total_ram = "Unknown"
        available_ram = "Unknown"

    gpu_info = "Not available"
    if CUDA_AVAILABLE:
        gpu_info = f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM"

    return cpu_count, total_ram, available_ram, gpu_info

def get_user_preference():
    print("\nResource Allocation Options:")
    print("1. CPU-only processing")
    print("2. GPU-assisted processing with CPU support")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice == '2'
        print("Invalid choice. Please enter 1 or 2.")

def is_unet_tensor(key, model_type):
    if model_type == "sd15":
        return key.startswith("model.diffusion_model.")
    elif model_type == "flux":
        return any(key.startswith(prefix) for prefix in [
            "unet.", "diffusion_model.", "model.diffusion_model.",
            "double_blocks.", "single_blocks.", "final_layer.",
            "guidance_in.", "img_in."
        ])
    elif model_type == "sdxl":
        return key.startswith("model.diffusion_model.")
    return False

def process_tensor(key, tensor, model_type, unet_tensors, non_unet_tensors, unet_count, verbose):
    if is_unet_tensor(key, model_type):
        if model_type == "sd15":
            new_key = key.replace("model.diffusion_model.", "")
            unet_tensors[new_key] = tensor.cpu()  # Move to CPU
        else:
            unet_tensors[key] = tensor.cpu()  # Move to CPU
        with unet_count.get_lock():
            unet_count.value += 1
        if verbose:
            logging.debug("Classified as UNet tensor")
    else:
        non_unet_tensors[key] = tensor.cpu()  # Move to CPU
        if verbose:
            logging.debug("Classified as non-UNet tensor")
    
    if verbose:
        logging.debug(f"Current UNet count: {unet_count.value}")
        logging.debug("---")

def save_tensors(tensors, output_file):
    try:
        save_file(tensors, output_file)
        logging.info(f"Successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving to {output_file}: {str(e)}")
        logging.debug(traceback.format_exc())
        raise

def process_model(input_file, unet_output_file, non_unet_output_file, model_type, use_gpu, verbose, num_threads, gpu_limit, cpu_limit):
    device = "cuda" if use_gpu and CUDA_AVAILABLE else "cpu"
    logging.info(f"Processing {input_file} on {device}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Using {num_threads} threads")
    if use_gpu:
        logging.info(f"GPU usage limit: {gpu_limit}%")
    logging.info(f"CPU usage limit: {cpu_limit}%")
    
    try:
        with safe_open(input_file, framework="pt", device=device) as f:
            unet_tensors = {}
            non_unet_tensors = {}
            total_tensors = 0
            unet_count = multiprocessing.Value('i', 0)
            key_prefixes = set()

            tensor_queue = queue.Queue()

            def worker():
                while True:
                    item = tensor_queue.get()
                    if item is None:
                        break
                    key, tensor = item
                    process_tensor(key, tensor, model_type, unet_tensors, non_unet_tensors, unet_count, verbose)
                    tensor_queue.task_done()
                    
                    # Implement CPU limiting
                    if cpu_limit < 100:
                        time.sleep((100 - cpu_limit) / 100 * 0.1)  # Adjust sleep time based on CPU limit

            threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            for key in f.keys():
                total_tensors += 1
                tensor = f.get_tensor(key)
                key_prefix = key.split('.')[0]
                key_prefixes.add(key_prefix)
                
                if verbose:
                    logging.debug(f"Processing key: {key}")
                    logging.debug(f"Tensor shape: {tensor.shape}")
                
                tensor_queue.put((key, tensor))

                # Implement GPU limiting
                if device == "cuda" and gpu_limit < 100:
                    current_gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    if current_gpu_usage > gpu_limit:
                        torch.cuda.empty_cache()
                        time.sleep(0.1)  # Allow some time for memory to be freed

            # Signal threads to exit
            for _ in range(num_threads):
                tensor_queue.put(None)

            # Wait for all threads to complete
            for t in threads:
                t.join()

            logging.info(f"Total tensors processed: {total_tensors}")
            logging.info(f"UNet tensors: {unet_count.value}")
            logging.info(f"Non-UNet tensors: {total_tensors - unet_count.value}")
            logging.info(f"Unique key prefixes found: {', '.join(sorted(key_prefixes))}")

        if unet_count.value == 0:
            logging.warning("No UNet tensors were identified. Please check if the model type is correct.")

        logging.info(f"Saving extracted UNet to {unet_output_file}")
        save_tensors(unet_tensors, unet_output_file)
        
        logging.info(f"Saving model without UNet to {non_unet_output_file}")
        save_tensors(non_unet_tensors, non_unet_output_file)
        
        logging.info("Processing complete!")
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        logging.debug(traceback.format_exc())
        raise
    finally:
        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Extract UNet and create a model without UNet from SafeTensors file for SD 1.5, SDXL, or FLUX")
    parser.add_argument("input_file", type=Path, help="Input SafeTensors file")
    parser.add_argument("unet_output_file", type=Path, help="Output SafeTensors file for UNet")
    parser.add_argument("non_unet_output_file", type=Path, help="Output SafeTensors file for model without UNet")
    parser.add_argument("--model_type", choices=["sd15", "flux", "sdxl"], required=True, help="Type of model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use for processing (default: auto-detect)")
    parser.add_argument("--gpu_limit", type=int, default=90, help="Limit GPU usage to this percentage (default: 90)")
    parser.add_argument("--cpu_limit", type=int, default=90, help="Limit CPU usage to this percentage (default: 90)")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    cpu_count, total_ram, available_ram, gpu_info = check_resources()
    print(f"\nSystem Resources:")
    print(f"CPU Cores: {cpu_count}")
    print(f"Total RAM: {total_ram}")
    print(f"Available RAM: {available_ram}")
    print(f"GPU: {gpu_info}")

    use_gpu = get_user_preference() if CUDA_AVAILABLE else False
    
    # Auto-detect number of threads if not specified
    if args.num_threads is None:
        args.num_threads = max(1, os.cpu_count() - 1)  # Leave one core free
    
    try:
        process_model(args.input_file, args.unet_output_file, args.non_unet_output_file, 
                      args.model_type, use_gpu, args.verbose, args.num_threads, 
                      args.gpu_limit, args.cpu_limit)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.debug(traceback.format_exc())
        exit(1)

if __name__ == "__main__":
    main()
