import argparse
import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import threading
import queue
import multiprocessing

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def check_cuda():
    if CUDA_AVAILABLE:
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("CUDA is not available. Using CPU.")

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

def process_model(input_file, unet_output_file, non_unet_output_file, model_type, use_cpu, verbose, num_threads):
    device = "cpu" if use_cpu or not CUDA_AVAILABLE else "cuda"
    logging.info(f"Processing {input_file} on {device}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Using {num_threads} threads")
    
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
        save_file(unet_tensors, unet_output_file)
        
        logging.info(f"Saving model without UNet to {non_unet_output_file}")
        save_file(non_unet_tensors, non_unet_output_file)
        
        logging.info("Processing complete!")
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
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
    parser.add_argument("--use_cpu", action="store_true", help="Force use of CPU even if CUDA is available")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use for processing (default: auto-detect)")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    check_cuda()
    
    # Auto-detect number of threads if not specified
    if args.num_threads is None:
        args.num_threads = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    
    try:
        process_model(args.input_file, args.unet_output_file, args.non_unet_output_file, args.model_type, args.use_cpu, args.verbose, args.num_threads)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
