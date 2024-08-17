## UNet Extractor and Remover for Stable Diffusion 1.5, SDXL, and FLUX
This Python script (UNetExtractor.py) processes SafeTensors files for Stable Diffusion 1.5 (SD 1.5), Stable Diffusion XL (SDXL), and FLUX models. It extracts the UNet into a separate file and creates a new file with the remaining model components (without the UNet).

## Why UNet Extraction?
Using UNets instead of full checkpoints can save a significant amount of disk space, especially for models that utilize large text encoders. This is particularly beneficial for models like FLUX, which has a large number of parameters. Here's why:

Space Efficiency: Full checkpoints bundle the UNet, CLIP, VAE, and text encoder together. By extracting the UNet, you can reuse the same text encoder for multiple models, saving gigabytes of space per additional model.
Flexibility: You can download the text encoder once and use it with multiple UNet models, reducing redundancy and saving space.

Practical Example: Multiple full checkpoints of large models like FLUX can quickly consume tens of gigabytes. Using extracted UNets instead can significantly reduce storage requirements.
Future-Proofing: As models continue to grow in complexity, the space-saving benefits of using UNets become even more significant.

This tool helps you extract UNets from full checkpoints, allowing you to take advantage of these space-saving benefits across SD 1.5, SDXL, and open-source FLUX models.

## Features

- This tool supports UNet extraction for open-source FLUX models, including:
FLUX Dev: A mid-range version with open weights for non-commercial use.
FLUX Schnell: A faster version optimized for lower-end GPUs.
- Extracts UNet tensors from SafeTensors files
- Creates a separate SafeTensors file with non-UNet components
- Saves the extracted UNet as a new SafeTensors file
- Command-line interface for easy use
- Optional CUDA support for faster processing on compatible GPUs

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

## Usage

Run the script from the command line with the following syntax:

```
python UNetExtractor.py <input_file> <unet_output_file> <non_unet_output_file> --model_type <sd15|sdxl|flux> [--use_cpu]
```

### Arguments

- `<input_file>`: Path to the input SafeTensors file (full model)
- `<unet_output_file>`: Path where the extracted UNet will be saved
- `<non_unet_output_file>`: Path where the model without UNet will be saved
- `--model_type`: Specify the model type, either `sd15` for Stable Diffusion 1.5, `sdxl` for Stable Diffusion XL, or `flux` for FLUX models
- `--use_cpu`: (Optional) Force CPU usage even if CUDA is available

### Examples

For Stable Diffusion 1.5 using CUDA (if available):
```
python UNetExtractor.py path/to/sd15_model.safetensors path/to/output_sd15_unet.safetensors path/to/output_sd15_non_unet.safetensors --model_type sd15
```

For Stable Diffusion XL using CUDA (if available):
```
python UNetExtractor.py path/to/sdxl_model.safetensors path/to/output_sdxl_unet.safetensors path/to/output_sdxl_non_unet.safetensors --model_type sdxl
```

For FLUX models using CUDA (if available):
```
python UNetExtractor.py path/to/flux_model.safetensors path/to/output_flux_unet.safetensors path/to/output_flux_non_unet.safetensors --model_type flux
```

For any model type using CPU (even if CUDA is available):
```
python UNetExtractor.py path/to/model.safetensors path/to/output_unet.safetensors path/to/output_non_unet.safetensors --model_type <sd15|sdxl|flux> --use_cpu
```

## How It Works

1. The script checks for CUDA availability (if PyTorch is installed) and uses it if present (unless `--use_cpu` is specified).
2. It opens the input SafeTensors file using the `safetensors` library.
3. The script iterates through all tensors in the file, separating UNet-related tensors from other tensors.
4. For SD 1.5 and FLUX models, it removes the "model.diffusion_model." prefix from UNet tensor keys.
5. For SDXL, it keeps the original key names for both UNet and non-UNet tensors.
6. The extracted UNet tensors are saved to a new SafeTensors file.
7. The remaining non-UNet tensors are saved to a separate SafeTensors file.

## Notes

- The script automatically uses CUDA if available and PyTorch is installed, which can significantly speed up the process for large models like FLUX.
- If you prefer to use CPU even when CUDA is available, use the `--use_cpu` flag.
- If PyTorch is not installed or CUDA is not available, the script will automatically use CPU processing.
- Ensure you have sufficient disk space to save both output files, especially for large models like FLUX.
- Processing large models may take some time, depending on your system's performance and whether CUDA is used.

## Notes

- The script automatically uses CUDA if available and PyTorch is installed, which can significantly speed up the process for large models.
- If you prefer to use CPU even when CUDA is available, use the `--use_cpu` flag.
- If PyTorch is not installed or CUDA is not available, the script will automatically use CPU processing.
- Ensure you have sufficient disk space to save both output files.
- Processing large models may take some time, depending on your system's performance and whether CUDA is used.

...

## Troubleshooting

If you encounter any issues:

1. Ensure you have the latest version of the `safetensors` library installed.
2. If using CUDA, make sure you have a compatible version of PyTorch installed.
3. Check that your input file is a valid SafeTensors file for the specified model type.
4. Make sure you have read permissions for the input file and write permissions for the output directory.
5. If you're having issues with CUDA, try running with the `--use_cpu` flag to see if it resolves the problem.
6. If you encounter any "module not found" errors, ensure all required libraries are installed.

### NumPy and PyTorch Compatibility Issues

If you encounter an error related to NumPy version compatibility with PyTorch, such as:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.1 as it may crash.
```

Follow these steps to resolve the issue:

1. Create a new virtual environment:
   ```
   python -m venv unet_extractor_env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     unet_extractor_env\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source unet_extractor_env/bin/activate
     ```

3. Install specific versions of the required libraries:
   ```
   pip install numpy==1.23.5 torch==2.0.1 safetensors==0.3.1
   ```

   Note: The versions above are examples. You may need to adjust them based on your CUDA version and system requirements.

4. If you're using CUDA, install the CUDA-enabled version of PyTorch:
   ```
   pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
   ```
   Replace `cu117` with your CUDA version (e.g., `cu116`, `cu118`) if different.

5. Try running the script again within this virtual environment.

If you continue to experience issues, you may need to compile PyTorch from source with a compatible version of NumPy. However, this is an advanced solution and should only be attempted if you're comfortable with building Python packages from source.

Remember to activate the virtual environment each time you want to run the script:

```
source unet_extractor_env/bin/activate  # On macOS/Linux
unet_extractor_env\Scripts\activate     # On Windows
```

If you're still encountering problems after trying these steps, please open an issue on the GitHub repository with details about your system configuration and the full error message.

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
