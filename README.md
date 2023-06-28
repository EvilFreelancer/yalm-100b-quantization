# YaLM-100B Quantization Experiment

This project aims to enable the execution of the YaLM-100B Language Model on consumer-grade hardware by implementing a
quantization and prune procedures. It utilizes the Megatron-LM framework (forked and modified by Yandex), Nvidia Apex
and includes additional scripts for model quantization and pruning, in order to optimize model size and inference speed.

Main goal is to optimize the model size and making it feasible to run this large-scale model on `home` computers (not
just on a cluster of industrial cards) with powerful GPUs, such as a single RTX 3090/4090 video card.

However, it's essential to note that the model's large number of hidden layers (10240) pose a significant challenge. The
Megatron-LM framework reserves a substantial amount of memory (approximately 200GB VRAM) due to these layers, which
prevents successful model execution even after quantization and pruning.

The most likely solution to this challenge involves implementing an algorithm that uses the GPT-2 transformer in
combination with `torch.load` to load each weight file separately. Each weight file would then be used in a synchronous
sequential run of a seed text, with the results displayed to the user upon completion. Another promising directions
involves optimizing the Megatron-LM framework to enable it to run not only on GPUs but also on CPUs, or altering the
logic of VRAM reservation to match the actual memory footprint of the weights.

## Whats inside

* `quantize.py` - script for quantization weights to 8/4/2 bits
* `quantize_prune.py` - script for quantization weights to 8/4/2 bits and pruning model
* `predict.py` - script for model inference
* `interactive.sh` - script for interactive model inference, it will call `predict.py` script with required arguments
* `pr.py` - experimental script for utilizing `torch.load` to load each weight file separately

## Installation

### Clone repository

Clone the repository using the following command:

```shell
git clone --recurse-submodules
``` 

It will clone the repository and all submodules (Megatron-LM, YaLM-100B and Apex).

### Downloading checkpoint

Download model weights and vocabulary.

```shell
bash YaLM-100B/download/download.sh
```

By default, weights will be downloaded to `./YaLM-100B/download/yalm100b_checkpoint/weights/`, and vocabulary will be
downloaded to `./YaLM-100B/download/yalm100b_checkpoint/vocab/`, as another option, you can clone our HF repo and pull
the checkpoint.

### Docker

Requirements:

* CUDA 11.7
* Docker
* Docker Compose
* Nvidia Container Toolkit
* Nvidia Runtime Docker

To set up the project using Docker, follow the steps below:

1. Make sure Docker and Docker Compose are installed on your system. You can download them from the official Docker
   website.
2. Navigate to the project root directory.
3. Build the Docker image:

    ```shell
    cp docker-compose.dist.yml docker-compose.yml 
    docker-compose build
    ```

4. Run the Docker container:

   ```shell
   docker-compose up -d
   ```

5. Log into the Docker container:

   ```shell
   docker-compose exec app bash
   ```

### Local setup

Requirements:

* Python 3.8 (or 3.9)
* PyTorch 1.13.1 (megatron-lm from YaLM-100B repo doesn't work with newer versions, because of `six` package)
* CUDA 11.7 (because of Nvidia repo, there is only PyTorch 1.13.1+cu117 is available for download)

To set up the project locally, follow the steps below:

1. Prepare environment

   ```shell
   python3.8 -m venv venv
   source venv/bin/activate
   ```

2. Install all required dependencies

   ```shell
   pip install --upgrade pip
   pip install packaging==23.0 torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   pip install sentencepiece==0.1.99 deepspeed==0.9.5 six==1.16.0
   pip install --no-cache-dir -r requirements.txt
   ```

3. Then install Nvidia Apex

   ```shell
   git clone https://github.com/NVIDIA/apex.git
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
   ```

## Roadmap

The project is currently at an intermediate stage. Here is an overview of the development progress:

* [x] Model quantization script
* [x] Model pruning script
* [x] Script for standard model execution using Megatron-LM
* [ ] Script for quantized model execution using PyTorch only
  * [ ] Implementation for loading model layers sequentially rather than all at once
  * [ ] Parsing results and displaying them to the user
* [ ] Tunes of Megatron-LM framework
  * [ ] Enable it to run on CPUs

  * [ ] Reduce VRAM reservation

The final points in the roadmap present significant areas for future development. Sequential loading of model layers
could provide a solution to the current memory limitation issue, allowing for the execution of larger models on hardware
with less available VRAM. Moreover, enabling execution without reliance on the Megatron-LM framework may provide
additional flexibility for model deployment and execution.

Contributions to the project to help achieve these roadmap goals are welcomed and appreciated.

## Links

* https://github.com/yandex/YaLM-100B
* https://github.com/NVIDIA/Megatron-LM
* https://github.com/NVIDIA/apex
* https://huggingface.co/yandex/yalm-100b
* https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6
* https://www.reddit.com/r/MachineLearning/comments/vivji3/p_yandex_open_sources_100b_large_language_model/
* https://www.reddit.com/r/MachineLearning/comments/vpn0r1/d_has_anyone_got_yalm100b_to_run/
* https://xailient.com/blog/4-popular-model-compression-techniques-explained/
