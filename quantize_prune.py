import os
import torch
import argparse


# def prune_weights(weights, ratio):
#     threshold = torch.quantile(torch.abs(weights), ratio) # RuntimeError: quantile() input tensor is too large
#     mask = torch.abs(weights) > threshold
#     return weights * mask

def prune_weights(weights, ratio):
    k = int(weights.numel() * ratio)
    threshold = torch.topk(torch.abs(weights.view(-1)), k, largest=False).values.max()
    mask = torch.abs(weights) > threshold
    return weights * mask

def main():
    parser = argparse.ArgumentParser(description='Quantize and prune YaLM-100B checkpoints.')
    parser.add_argument('input_dir', type=str, help='Directory of original checkpoints.')
    parser.add_argument('output_dir', type=str, help='Directory to save quantized checkpoints.')
    parser.add_argument('bits', type=int, default=8, choices=[16, 8, 4, 2], help='Bit depth for quantization.')
    parser.add_argument('prune_ratio', type=float, default=0.2, help='Pruning ratio.')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist.")
        exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Quantization bit depth
    bits = args.bits
    if bits == 16:
        print("16-bit quantization is not supported.")
        return
    elif bits == 8:
        dtype = torch.quint8
    elif bits == 4:
        dtype = torch.quint4x2
    elif bits == 2:
        dtype = torch.quint2x4
    else:
        print(f"Quantization with {bits} bits is not supported.")
        return

    files = sorted(os.listdir(args.input_dir))
    num_files = len(files)
    print(f"Processing {num_files} files...")

    for i, filename in enumerate(files, 1):
        print(f"Processing file {i} of {num_files}: {filename}")
        file_path = os.path.join(args.input_dir, filename)
        weights_dict = torch.load(file_path)
        quantized_weights_dict = {}
        for param_name, weights in weights_dict.items():
            if isinstance(weights, torch.Tensor):
                # Prune the weights
                weights = prune_weights(weights, args.prune_ratio)
                # Quantize only works on Float Tensor, by default BFloat16
                weights = weights.float()
                quantized_weights_dict[param_name] = torch.quantize_per_tensor(
                    weights,
                    scale=0.01,
                    zero_point=0,
                    dtype=dtype
                )
        output_path = os.path.join(args.output_dir, filename)
        torch.save(quantized_weights_dict, output_path)
        print(f"Saved quantized weights to {output_path}")


if __name__ == "__main__":
    main()
