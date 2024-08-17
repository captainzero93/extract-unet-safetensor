import argparse
from safetensors import safe_open
from safetensors.torch import save_file

def process_model(input_file, unet_output_file, non_unet_output_file, model_type):
    print(f"Processing {input_file}")
    
    with safe_open(input_file, framework="pt", device="cpu") as f:
        unet_tensors = {}
        non_unet_tensors = {}
        for key in f.keys():
            if model_type == "sd15":
                if key.startswith("model.diffusion_model."):
                    new_key = key.replace("model.diffusion_model.", "")
                    unet_tensors[new_key] = f.get_tensor(key)
                else:
                    non_unet_tensors[key] = f.get_tensor(key)
            elif model_type == "sdxl":
                if key.startswith("model.diffusion_model."):
                    unet_tensors[key] = f.get_tensor(key)
                else:
                    non_unet_tensors[key] = f.get_tensor(key)

    print(f"Saving extracted UNet to {unet_output_file}")
    save_file(unet_tensors, unet_output_file)
    
    print(f"Saving model without UNet to {non_unet_output_file}")
    save_file(non_unet_tensors, non_unet_output_file)
    
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Extract UNet and create a model without UNet from SafeTensors file for SD 1.5 or SDXL")
    parser.add_argument("input_file", help="Input SafeTensors file")
    parser.add_argument("unet_output_file", help="Output SafeTensors file for UNet")
    parser.add_argument("non_unet_output_file", help="Output SafeTensors file for model without UNet")
    parser.add_argument("--model_type", choices=["sd15", "sdxl"], required=True, help="Model type: sd15 or sdxl")
    
    args = parser.parse_args()
    
    process_model(args.input_file, args.unet_output_file, args.non_unet_output_file, args.model_type)

if __name__ == "__main__":
    main()
