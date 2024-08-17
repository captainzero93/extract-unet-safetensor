# UNet Extractor and Remover for Stable Diffusion 1.5, SDXL, and FLUX

This Python script (UNetExtractor.py) processes SafeTensors files for Stable Diffusion 1.5 (SD 1.5), Stable Diffusion XL (SDXL), and FLUX models. It extracts the UNet into a separate file and creates a new file with the remaining model components (without the UNet).

## Why UNet Extraction?

Using UNets instead of full checkpoints can save a significant amount of disk space, especially for models that utilize large text encoders. This is particularly beneficial for models like FLUX, which has a large number of parameters. Here's why:

- Space Efficiency: Full checkpoints bundle the UNet, CLIP, VAE, and text encoder together. By extracting the UNet, you can reuse the same text encoder for multiple models, saving gigabytes of space per additional model.
- Flexibility: You can download the text encoder once and use it with multiple UNet models, reducing redundancy and saving space.
- Practical Example: Multiple full checkpoints of large models like FLUX can quickly consume tens of gigabytes. Using extracted UNets instead can significantly reduce storage requirements.
- Future-Proofing: As models continue to grow in complexity, the space-saving benefits of using UNets become even more significant.

This tool helps you extract UNets from full checkpoints, allowing you to take advantage of these space-saving benefits across SD 1.5, SDXL, and open-source FLUX models.

## Features

- Supports UNet extraction for SD 1.5, SDXL, and open-source FLUX models, including:
  - FLUX Dev: A mid-range version with open weights for non-commercial use.
  - FLUX Schnell: A faster version optimized for lower-end GPUs.
- Extracts UNet tensors from SafeTensors files
- Creates a separate SafeTensors file with non-UNet components
- Saves the extracted UNet as a new SafeTensors file
- Command-line interface for easy use
- Optional CUDA support for faster processing on compatible GPUs
- Automatic thread detection for optimal CPU usage
- Improved memory management with RAM offloading
- Multi-threading support for faster processing

## Requirements

- Python 3.6+
- safetensors library
- PyTorch (optional, for CUDA support)

## Installation

1. Clone this repository or download the `UNetExtractor.py` script.

2. Install the required libraries:

   ```
   pip install safetensors
   ```

3. Optionally, if you want CUDA support, install PyTorch:

   ```
   pip install torch torchvision torchaudio
   ```

Optional: Install psutil for enhanced system resource reporting
   ```
pip install psutil
   ```

## Usage

Run the script from the command line with the following syntax:

```
python UNetExtractor.py <input_file> <unet_output_file> <non_unet_output_file> --model_type <sd15|sdxl|flux> [--use_cpu] [--verbose] [--num_threads <num>]
```

### Arguments

- `<input_file>`: Path to the input SafeTensors file (full model)
- `<unet_output_file>`: Path where the extracted UNet will be saved
- `<non_unet_output_file>`: Path where the model without UNet will be saved
- `--model_type`: Specify the model type, either `sd15` for Stable Diffusion 1.5, `sdxl` for Stable Diffusion XL, or `flux` for FLUX models
- `--use_cpu`: (Optional) Force CPU usage even if CUDA is available
- `--verbose`: (Optional) Enable verbose logging for detailed process information
- `--num_threads`: (Optional) Specify the number of threads to use for processing. If not specified, the script will automatically detect the optimal number of threads.

### Examples

For Stable Diffusion 1.5 using CUDA (if available):
```
python UNetExtractor.py path/to/sd15_model.safetensors path/to/output_sd15_unet.safetensors path/to/output_sd15_non_unet.safetensors --model_type sd15 --verbose
```

For Stable Diffusion XL using CUDA (if available):
```
python UNetExtractor.py path/to/sdxl_model.safetensors path/to/output_sdxl_unet.safetensors path/to/output_sdxl_non_unet.safetensors --model_type sdxl --verbose
```

For FLUX models using CUDA (if available) with 8 threads:
```
python UNetExtractor.py path/to/flux_model.safetensors path/to/output_flux_unet.safetensors path/to/output_flux_non_unet.safetensors --model_type flux --verbose --num_threads 8
```

For any model type using CPU (even if CUDA is available):
```
python UNetExtractor.py path/to/model.safetensors path/to/output_unet.safetensors path/to/output_non_unet.safetensors --model_type <sd15|sdxl|flux> --use_cpu --verbose
```

## How It Works

1. The script checks for CUDA availability (if PyTorch is installed) and uses it if present (unless `--use_cpu` is specified).
2. It determines the optimal number of threads to use based on the system's CPU cores (if not manually specified).
3. It opens the input SafeTensors file using the `safetensors` library.
4. The script iterates through all tensors in the file, separating UNet-related tensors from other tensors.
5. For SD 1.5 and FLUX models, it removes the "model.diffusion_model." prefix from UNet tensor keys.
6. For SDXL, it keeps the original key names for both UNet and non-UNet tensors.
7. The extracted UNet tensors are saved to a new SafeTensors file.
8. The remaining non-UNet tensors are saved to a separate SafeTensors file.
9. The script uses multi-threading to process tensors concurrently, improving performance.
10. RAM offloading is implemented to manage memory usage, especially for large models.

## Notes

- The script automatically uses CUDA if available and PyTorch is installed, which can significantly speed up the process for large models like FLUX.
- If you prefer to use CPU even when CUDA is available, use the `--use_cpu` flag.
- The script automatically detects the optimal number of threads to use, but you can manually specify this with `--num_threads`.
- If PyTorch is not installed or CUDA is not available, the script will automatically use CPU processing.
- Ensure you have sufficient disk space to save both output files, especially for large models like FLUX.
- Processing large models may take some time, depending on your system's performance, whether CUDA is used, and the number of threads employed.

## Troubleshooting

[The troubleshooting section remains the same as in your original README]

## Contributing

Contributions, issues, and feature requests are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use UNet Extractor and Remover in your research or projects, please cite it as follows:

```
[Joe Faulkner] (captainzero93). (2024). UNet Extractor and Remover for Stable Diffusion 1.5, SDXL, and FLUX. GitHub. https://github.com/captainzero93/unet-extractor
```

## Acknowledgements

- This script uses the `safetensors` library developed by the Hugging Face team.
- Inspired by Stable Diffusion and the community.
