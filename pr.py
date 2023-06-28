import torch
import os
import sentencepiece as spm
from time import sleep
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

ROOT = "data"
VOCAB_PATH = ROOT + "/vocab/voc_100b.sp"
MODEL_PATH = ROOT + "/weights"
LAYER_PREFIX = "layer_"
INPUT_TEXT = "Hello, world!"

vocab_size = 128000
hidden_size = 2048
num_attention_heads = 16

# Load sentencepiece model
sp = spm.SentencePieceProcessor()
sp.Load(VOCAB_PATH)


class MyGPT2Model(nn.Module):
    def __init__(self, config):
        super(MyGPT2Model, self).__init__()
        self.config = config
        self.transformer = GPT2Model(config)

    def forward(self, input_ids, past_key_values=None):
        return self.transformer(input_ids, past_key_values=past_key_values)

    def load_weights_for_layer(self, layer_num, weights):
        self.transformer.h[layer_num].load_state_dict(weights)


def load_model_by_parts(model_path, layer_prefix):
    """
    Load model by parts.

    :param model_path: Path to the model.
    :param layer_prefix: Prefix of the layer files.
    :return: A dictionary with model parts.
    """
    model_parts = {}

    # Iterate over files in model directory
    files = sorted(os.listdir(model_path))
    for file_name in files:
        if file_name.startswith(layer_prefix):
            print("Loading layer: " + file_name)
            layer_path = os.path.join(model_path, file_name)
            model_parts[file_name] = torch.load(layer_path, map_location='cpu')

    return model_parts


def process_model_parts(model_parts, input_text):
    """
    Process model parts.

    :param model_parts: A dictionary with model parts.
    :param input_text: Input text for the model.
    :return: Result of the model.
    """
    # Convert input text to tokens
    input_tokens = sp.EncodeAsIds(input_text)

    # Convert tokens to tensor
    input_tensor = torch.tensor([input_tokens], dtype=torch.long)

    # Output tensor
    output_tensor = input_tensor

    for part_name, part in model_parts.items():
        # Apply part to output tensor
        output_tensor = part(output_tensor)

    # Convert output tensor to tokens
    output_tokens = output_tensor[0].tolist()

    # print(output_tokens)
    # exit(1)

    # Convert tokens to text
    return sp.DecodeIds(output_tokens)


def load_and_apply_weights(model, weights_dir):
    layer_files = sorted(os.listdir(weights_dir))
    input_ids = torch.tensor([[1, 2, 3, 4]])  # Example input

    for i, layer_file in enumerate(layer_files):
        print(f"Loading weights for layer {i} from {layer_file}")
        weights = torch.load(os.path.join(weights_dir, layer_file))

        # dequantize weights
        for key, weight in weights.items():
            if weight.is_quantized:
                weights[key] = weight.dequantize()

        if 'word_embeddings.weight' in weights:
            print("Loading word embeddings")
            model.transformer.wte.load_state_dict({'weight': weights['word_embeddings.weight']})
            del weights['word_embeddings.weight']

        # print(weights)
        # exit()
        model.load_weights_for_layer(i, weights)

        print(f"Applying weights to input data")
        outputs = model(input_ids)
        input_ids = outputs.last_hidden_state.argmax(-1)

    return input_ids


# Load model by parts
# model_parts = load_model_by_parts(MODEL_PATH, LAYER_PREFIX)
config = GPT2Config(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_attention_heads=num_attention_heads,
)
model = MyGPT2Model(config)
final_ids = load_and_apply_weights(model, MODEL_PATH)

# Process model parts
# output_text = process_model_parts(model_parts, INPUT_TEXT)

# Print output text
# print(output_text)
