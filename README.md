# UNet Extractor for Stable Diffusion 1.5 and SDXL

This Python script extracts the UNet from SafeTensors files for Stable Diffusion 1.5 (SD 1.5) and Stable Diffusion XL (SDXL) models. It allows you to easily separate the UNet component from a full model file, which can be useful for various purposes such as model merging, fine-tuning, or analysis.

## Features

- Supports both SD 1.5 and SDXL model architectures
- Extracts UNet tensors from SafeTensors files
- Saves the extracted UNet as a new SafeTensors file
- Command-line interface for easy use

## Requirements

- Python 3.6+
- safetensors library

## Installation

1. Clone this repository or download the `unet_extractor.py` script.

2. Install the required `safetensors` library:

   ```
   pip install safetensors
   ```

## Usage

Run the script from the command line with the following syntax:

```
python unet_extractor.py <input_file> <output_file> --model_type <sd15|sdxl>
```

### Arguments

- `<input_file>`: Path to the input SafeTensors file (full model)
- `<output_file>`: Path where the extracted UNet will be saved
- `--model_type`: Specify the model type, either `sd15` for Stable Diffusion 1.5 or `sdxl` for Stable Diffusion XL

### Examples

For Stable Diffusion 1.5:
```
python unet_extractor.py path/to/sd15_model.safetensors path/to/output_sd15_unet.safetensors --model_type sd15
```

For Stable Diffusion XL:
```
python unet_extractor.py path/to/sdxl_model.safetensors path/to/output_sdxl_unet.safetensors --model_type sdxl
```

## How It Works

1. The script opens the input SafeTensors file using the `safetensors` library.
2. It iterates through all tensors in the file, identifying UNet-related tensors based on their key names.
3. For SD 1.5, it removes the "model.diffusion_model." prefix from tensor keys.
4. For SDXL, it keeps the original key names.
5. The extracted UNet tensors are saved to a new SafeTensors file.

## Notes

- Ensure you have sufficient disk space to save the extracted UNet file.
- The script processes the tensors in CPU memory, so it should work even on machines without a GPU.
- Processing large models may take some time, depending on your system's performance.

## Troubleshooting

If you encounter any issues:

1. Ensure you have the latest version of the `safetensors` library installed.
2. Check that your input file is a valid SafeTensors file for the specified model type.
3. Make sure you have read permissions for the input file and write permissions for the output directory.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](link-to-your-issues-page) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](link-to-your-license-file) file for details.

## Acknowledgements

- This script uses the `safetensors` library developed by the Hugging Face team.
- Inspired by the Stable Diffusion and SDXL projects from Stability AI.
- DBacon1052 and BlastedRemnants on Reddit for posting about the idea https://www.reddit.com/r/StableDiffusion/comments/1eu4idh/comment/liic8fj/

