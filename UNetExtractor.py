import argparse
from safetensors import safe_open
from safetensors.torch import save_file

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

def process_model(input_file, unet_output_file, non_unet_output_file, model_type, use_cpu):
    device = "cpu" if use_cpu or not CUDA_AVAILABLE else "cuda"
    print(f"Processing {input_file} on {device}")
    
    with safe_open(input_file, framework="pt", device=device) as f:
        unet_tensors = {}
        non_unet_tensors = {}
        total_tensors = 0
        unet_count = 0

        for key in f.keys():
            total_tensors += 1
            tensor = f.get_tensor(key)
            
            if model_type == "sd15":
                if key.startswith("model.diffusion_model."):
                    new_key = key.replace("model.diffusion_model.", "")
                    unet_tensors[new_key] = tensor
                    unet_count += 1
                else:
                    non_unet_tensors[key] = tensor
            elif model_type == "flux":
                if any(key.startswith(prefix) for prefix in ["unet.", "diffusion_model.", "model.diffusion_model."]):
                    unet_tensors[key] = tensor
                    unet_count += 1
                else:
                    non_unet_tensors[key] = tensor
            elif model_type == "sdxl":
                if key.startswith("model.diffusion_model."):
                    unet_tensors[key] = tensor
                    unet_count += 1
                else:
                    non_unet_tensors[key] = tensor

        print(f"Total tensors processed: {total_tensors}")
        print(f"UNet tensors: {unet_count}")
        print(f"Non-UNet tensors: {total_tensors - unet_count}")

    print(f"Saving extracted UNet to {unet_output_file}")
    save_file(unet_tensors, unet_output_file)
    
    print(f"Saving model without UNet to {non_unet_output_file}")
    save_file(non_unet_tensors, non_unet_output_file)
    
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Extract UNet and create a model without UNet from SafeTensors file for SD 1.5, SDXL, or FLUX")
    parser.add_argument("input_file", help="Input SafeTensors file")
    parser.add_argument("unet_output_file", help="Output SafeTensors file for UNet")
    parser.add_argument("non_unet_output_file", help="Output SafeTensors file for model without UNet")
    parser.add_argument("--model_type", choices=["sd15", "sdxl", "flux"], required=True, help="Model type: sd15, sdxl, or flux")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    
    args = parser.parse_args()
    
    process_model(args.input_file, args.unet_output_file, args.non_unet_output_file, args.model_type, args.use_cpu)

if __name__ == "__main__":
    main()
