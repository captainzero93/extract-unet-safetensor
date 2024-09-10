# UNet Extractor and Remover for Stable Diffusion 1.5, SDXL, and FLUX

This Python script (UNetExtractor.py) processes SafeTensors files for Stable Diffusion 1.5 (SD 1.5), Stable Diffusion XL (SDXL), and FLUX models. It extracts the UNet into a separate file and creates a new file with the remaining model components (without the UNet).

![FLUX Example](https://raw.githubusercontent.com/captainzero93/extract-unet-safetensor/main/fluxeample.png)

Above example: UNetExtractor.py flux1-dev.safetensors flux1-dev_unet.safetensors flux1-dev_non_unet.safetensors --model_type flux --verbose

## AUTOMATIC1111 Extension for UNet Loading

We've developed an extension for AUTOMATIC1111's Stable Diffusion Web UI that allows you to load and use the extracted UNet files directly within the interface. This extension seamlessly integrates with the txt2img workflow, enabling you to utilize the space-saving benefits of separated UNet files without compromising on functionality.

### Extension Features:
- Load separate UNet and non-UNet files
- Combine them on-the-fly for use in generation
- Compatible with files created by this UNet Extractor tool
- Integrated into the AUTOMATIC1111 Web UI for easy use

To use the extension, please visit our [UNet Loader Extension Repository](https://github.com/captainzero93/load-extracted-unet-automatic1111) for installation and usage instructions.

## Why UNet Extraction?

Using UNets instead of full checkpoints can save a significant amount of disk space, especially for models that utilize large text encoders. This is particularly beneficial for models like FLUX, which has a large number of parameters. Here's why:

- Space Efficiency: Full checkpoints bundle the UNet, CLIP, VAE, and text encoder together. By extracting the UNet, you can reuse the same text encoder for multiple models, saving gigabytes of space per additional model.
- Flexibility: You can download the text encoder once and use it with multiple UNet models, reducing redundancy and saving space.
- Practical Example: Multiple full checkpoints of large models like FLUX can quickly consume tens of gigabytes. Using extracted UNets instead can significantly reduce storage requirements.
- Future-Proofing: As models continue to grow in complexity, the space-saving benefits of using UNets (may) become even more significant.

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
- User choice between CPU-only and GPU-assisted processing
- GPU and CPU usage limiting options
- Enhanced error handling and logging
- Detailed debugging information for troubleshooting
- AUTOMATIC1111 extension for seamless integration with Stable Diffusion Web UI

## Requirements

- Python 3.6+
- safetensors library
- PyTorch (optional, for CUDA support)
- psutil (optional, for enhanced system resource reporting)

## Installation

1. Clone this repository or download the `UNetExtractor.py` script.

2. It's recommended to create a new virtual environment:
   ```
   python -m venv unet_extractor_env
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     unet_extractor_env\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source unet_extractor_env/bin/activate
     ```

4. Install the required libraries with specific versions for debugging:

   ```
   pip install numpy==1.23.5 torch==2.0.1 safetensors==0.3.1
   ```

5. If you're using CUDA, install the CUDA-enabled version of PyTorch:
   ```
   pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
   ```
   Replace `cu117` with your CUDA version (e.g., `cu116`, `cu118`) if different.

6. Optionally, install psutil for enhanced system resource reporting:
   ```
   pip install psutil==5.9.0
   ```

Note: The versions above are examples and may need to be adjusted based on your system requirements and CUDA version. These specific versions are recommended for debugging purposes as they are known to work together. For regular use, you may use the latest versions of these libraries.

## Usage

Run the script from the command line with the following syntax:

```
python UNetExtractor.py <input_file> <unet_output_file> <non_unet_output_file> --model_type <sd15|sdxl|flux> [--verbose] [--num_threads <num>] [--gpu_limit <percent>] [--cpu_limit <percent>]
```

### Arguments

- `<input_file>`: Path to the input SafeTensors file (full model)
- `<unet_output_file>`: Path where the extracted UNet will be saved
- `<non_unet_output_file>`: Path where the model without UNet will be saved
- `--model_type`: Specify the model type, either `sd15` for Stable Diffusion 1.5, `sdxl` for Stable Diffusion XL, or `flux` for FLUX models
- `--verbose`: (Optional) Enable verbose logging for detailed process information
- `--num_threads`: (Optional) Specify the number of threads to use for processing. If not specified, the script will automatically detect the optimal number of threads.
- `--gpu_limit`: (Optional) Limit GPU usage to this percentage (default: 90)
- `--cpu_limit`: (Optional) Limit CPU usage to this percentage (default: 90)

### Examples

For Stable Diffusion 1.5 using CUDA (if available):
```
python UNetExtractor.py path/to/sd15_model.safetensors path/to/output_sd15_unet.safetensors path/to/output_sd15_non_unet.safetensors --model_type sd15 --verbose
```

For Stable Diffusion XL using CUDA (if available):
```
python UNetExtractor.py path/to/sdxl_model.safetensors path/to/output_sdxl_unet.safetensors path/to/output_sdxl_non_unet.safetensors --model_type sdxl --verbose
```

For FLUX models using CUDA (if available) with 8 threads and 80% GPU usage limit:
```
python UNetExtractor.py path/to/flux_model.safetensors path/to/output_flux_unet.safetensors path/to/output_flux_non_unet.safetensors --model_type flux --verbose --num_threads 8 --gpu_limit 80
```

## How It Works

1. The script checks for CUDA availability (if PyTorch is installed) and prompts to choose between CPU-only and GPU-assisted processing.
2. It determines the optimal number of threads to use based on the system's CPU cores (if not manually specified).
3. It opens the input SafeTensors file using the `safetensors` library.
4. The script iterates through all tensors in the file, separating UNet-related tensors from other tensors.
5. For SD 1.5 and FLUX models, it removes the "model.diffusion_model." prefix from UNet tensor keys.
6. For SDXL, it keeps the original key names for both UNet and non-UNet tensors.
7. The script uses multi-threading to process tensors concurrently, improving performance.
8. GPU and CPU usage are limited based on user-specified percentages or default values.
9. The extracted UNet tensors are saved to a new SafeTensors file.
10. The remaining non-UNet tensors are saved to a separate SafeTensors file.
11. RAM offloading is implemented to manage memory usage, especially for large models.

## Using Extracted UNets with AUTOMATIC1111

After extracting UNet files using this tool, you can easily use them in AUTOMATIC1111's Stable Diffusion Web UI:

1. Install our UNet Loader extension in your AUTOMATIC1111 setup.
2. Place the extracted UNet and non-UNet files in the extension's designated folder.
3. Use the extension's interface to select and load your desired UNet and non-UNet components.
4. Generate images using txt2img as usual, now benefiting from the space-saving and flexibility of separated UNet files.

For detailed instructions, please refer to the [UNet Loader Extension Repository](https://github.com/captainzero93/unet-loader-extension).

## Debugging Information

When running the script with the `--verbose` flag, you'll see detailed debugging information, including:

- System resource information (CPU cores, RAM, GPU)
- Processing details for each tensor (key, shape, classification)
- Running count of UNet and non-UNet tensors
- GPU memory usage (if applicable)
- Detailed error messages and stack traces in case of exceptions

Example debug output:

```
2024-08-17 21:06:30,500 - DEBUG - Current UNet count: 770
2024-08-17 21:06:30,500 - DEBUG - ---
2024-08-17 21:06:31,142 - DEBUG - Processing key: vector_in.out_layer.weight
2024-08-17 21:06:31,142 - DEBUG - Tensor shape: torch.Size([3072, 3072])
2024-08-17 21:06:31,172 - DEBUG - Classified as non-UNet tensor
2024-08-17 21:06:31,172 - DEBUG - Current UNet count: 770
2024-08-17 21:06:31,172 - DEBUG - ---
2024-08-17 21:06:31,203 - INFO - Total tensors processed: 780
2024-08-17 21:06:31,203 - INFO - UNet tensors: 770
2024-08-17 21:06:31,203 - INFO - Non-UNet tensors: 10
2024-08-17 21:06:31,203 - INFO - Unique key prefixes found: double_blocks, final_layer, guidance_in, img_in, single_blocks, time_in, txt_in, vector_in
```

This output helps identify issues with tensor classification, resource usage, and overall processing flow.

## Notes

- The script now prompts the user to choose between CPU-only and GPU-assisted processing if CUDA is available.
- Automatic thread detection is used if the number of threads is not specified.
- GPU and CPU usage can be limited to prevent normal system use slowdowns during processing.
- Enhanced error handling and logging provide more informative output during processing.
- The disk space check has been removed to avoid potential errors on some systems.

## Troubleshooting

If you encounter any issues:

1. Ensure you're using the recommended library versions as specified in the Installation section.
2. Run the script with the `--verbose` flag to get detailed debugging information.
3. Check for compatibility between your CUDA version and the installed PyTorch version.
4. If you encounter a NumPy version compatibility error with PyTorch, such as:
   ```
   A module that was compiled using NumPy 1.x cannot be run in
   NumPy 2.0.1 as it may crash.
   ```
   Try downgrading NumPy to version 1.23.5 as recommended in the installation instructions.
5. Ensure you have the latest version of the `safetensors` library installed.
6. Check that your input file is a valid SafeTensors file for the specified model type.
7. Make sure you have read permissions for the input file and write permissions for the output directory.
8. If you're having issues with CUDA, try running with CPU-only processing to see if it resolves the problem.
9. If you encounter any "module not found" errors, ensure all required libraries are installed in your virtual environment.
10. Check the console output for any error messages or stack traces that can help identify the issue.

If you continue to experience issues after trying these steps, please open an issue on the GitHub repository with details about your system configuration, the command you're using, and the full error message or debugging output.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/captainzero93/extract-unet-safetensor/issues) if you want to contribute.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use UNet Extractor and Remover in your research or projects, please cite it as follows:

```
[Joe Faulkner] (captainzero93). (2024). UNet Extractor and Remover for Stable Diffusion 1.5, SDXL, and FLUX. GitHub. https://github.com/captainzero93/unet-extractor
```

## Acknowledgements

- This script uses the `safetensors` library developed by the Hugging Face team.
- Inspired by Stable Diffusion and the FLUX model community.
- Special thanks to all contributors and users who have provided feedback and suggestions.
- u/DBacon1052 
- u/BlastedRemnants/
- rauldlnx10 on GitHub for the Forge extension.
- All users and contributors of the AUTOMATIC1111 Stable Diffusion Web UI community.
