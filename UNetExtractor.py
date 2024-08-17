import argparse
from safetensors import safe_open
from safetensors.torch import save_file

def extract_unet(input_file, output_file, model_type):
    print(f"Extracting UNet from {input_file}")
    
    with safe_open(input_file, framework="pt", device="cpu") as f:
        unet_tensors = {}
        for key in f.keys():
            if model_type == "sd15" and key.startswith("model.diffusion_model."):
                new_key = key.replace("model.diffusion_model.", "")
                unet_tensors[new_key] = f.get_tensor(key)
            elif model_type == "sdxl" and key.startswith("model.diffusion_model."):
                unet_tensors[key] = f.get_tensor(key)

    print(f"Saving extracted UNet to {output_file}")
    save_file(unet_tensors, output_file)
    print("Extraction complete!")

def main():
    parser = argparse.ArgumentParser(description="Extract UNet from SafeTensors file for SD 1.5 or SDXL")
    parser.add_argument("input_file", help="Input SafeTensors file")
    parser.add_argument("output_file", help="Output SafeTensors file for UNet")
    parser.add_argument("--model_type", choices=["sd15", "sdxl"], required=True, help="Model type: sd15 or sdxl")
    
    args = parser.parse_args()
    
    extract_unet(args.input_file, args.output_file, args.model_type)

if __name__ == "__main__":
    main()
