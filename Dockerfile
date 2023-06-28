#FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:23.03-py3
#ENV PATH="/root/.cargo/bin:${PATH}"
#ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install required packages
RUN set -xe \
 && apt-get -y update \
 && apt-get install -fyqq software-properties-common curl build-essential git \
 && apt-get -y update \
 && add-apt-repository universe \
 && apt-get -y update \
 && apt-get -fyqq install python3 python3-pip \
 && apt-get clean

# Install Rust
#RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# Install python packages
COPY requirements.txt ./
RUN set -xe \
 && pip install --upgrade pip \
 && pip install sentencepiece==0.1.99 deepspeed==0.9.5 six==1.16.0 \
 && pip install --no-cache-dir -r requirements.txt

# Install apex (already in nvcr.io/nvidia/pytorch image)
#RUN set -xe  \
# && git clone https://github.com/NVIDIA/apex.git \
# && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# Copy project files
COPY . .

CMD ["sleep", "inf"]
