# Hymba


<p align="center">
        🤗 <a href="https://huggingface.co/collections/nvidia/hymba-673c35516c12c4b98b5e845f">Hugging Face Models</a>&nbsp&nbsp | &nbsp&nbsp 📄 <a href="xxx">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📜 <a href="xxx">Blog</a> &nbsp </a>
</p>

## Introduction

Hymba is a family of small language models (SLMs) featuring a hybrid-head parallel architecture that integrates transformer attention mechanisms with SSMs to achieve the best of both worlds: enhanced efficiency and improved performance. In Hymba, attention heads provide high-resolution recall, while SSM heads enable efficient context summarization.

## News

1. 🚀 Hymba is out! Check our blog post: [Hymba: A Hybrid-head Architecture for Small Language Models.](xxx).

## Hymba Model Performance

<p align="center">
  <img src="images/hymba-performance.png" alt="Hymba Performance" width="512"/>
  <p align="center">Performance comparison of Hymba-1.5B-Base against sub-2B models in terms of average accuracy, cache size (MB) relative to sequence length, and throughput (tok/sec).</p>
</p>

As shown in the above figure, we compare Hymba-1.5B against sub-2B models (LlaMA 3.2-1B, OpenELM-1B, Phi-1.5, SmolLM2-1.7B, danube2-1.8B, Qwen2.5-1.5B) in terms of average task accuracy, cache size (MB) relative to sequence length, and throughput (tok/sec).
In this set of experiments, the tasks include MMLU, ARC-C, ARC-E, PIQA, Hellaswag, Winogrande, and SQuAD-C, and the throughput is measured on an NVIDIA A100 with a sequence length of 8k and a batch size of 128 using PyTorch. 
For models encountering out-of-memory (OOM) issues during throughput measurement, we halve the batch size until the OOM is resolved to measure the maximal achievable throughput without OOM.

## Hugging Face Checkpoints, Model Cards and Usage

Please see:

1. [Hymba-1.5B-Base](https://huggingface.co/nvidia/Hymba-1.5B-Base)
2. [Hymba-1.5B-Instruct](https://huggingface.co/nvidia/Hymba-1.5B-Instruct)



## Model Usage

### Environment Setup

Since our model employs [FlexAttention](https://pytorch.org/blog/flexattention/), which relies on Pytorch 2.5 and other related dependencies, we provide three ways to set up the environment:

- **[Local Install]** Install the related packages using our provided `setup.sh` (support CUDA 12.1/12.4):
```
wget --header="Authorization: Bearer YOUR_HF_TOKEN" https://huggingface.co/nvidia/Hymba-1.5B-Instruct/resolve/main/setup.sh
bash setup.sh
```

- **[Docker]** We have prepared a docker image with all of Hymba's dependencies installed. You can download our docker image and start a container using the following commands:

```
wget http://10.137.9.244:8000/hymba_docker.tar
docker load -i hymba.tar
docker run --security-opt seccomp=unconfined --gpus all -v /home/$USER:/home/$USER -it hymba:v1 bash
```

### Chat with Hymba
After setting up the environment, you can use the following script to chat with our model in the command line.

```
python chat.py
```

### Finetuning Hymba


[LMFlow](https://github.com/OptimalScale/LMFlow) is a complete pipeline for fine-tuning large language models. 
The following steps provide an example of how to fine-tune the `Hymba-1.5B-Base` models using LMFlow with the `alpaca` dataset.

1. Install LMFlow

    ```
    git clone https://github.com/OptimalScale/LMFlow.git
    cd LMFlow
    bash install.sh
    ```
  
2. Prepare the dataset
  
      Download the [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset and preprocess it using the following command.
  
      ```bash
      cd data && ./download.sh alpaca && cd -
      ```

3. Fine-tune the model
  
      Fine-tune the Hymba-1.5B-Base model on the alpaca dataset using the following command.
    
      ```bash
      bash ./scripts/run_finetune.sh \
        --model_name_or_path nvidia/Hymba-1.5B-Base \
        --dataset_path data/alpaca/train_conversation \
        --output_model_path output_models/finetuned_hymba
      ```

With LMFlow, you can also fine-tune the model on your custom dataset. The only thing you need to do is transform your dataset into the [LMFlow data format](https://optimalscale.github.io/LMFlow/examples/DATASETS.html).
In addition to full-finetuniing, you can also fine-tune hymba efficiently with [DoRA](https://arxiv.org/html/2402.09353v4), [LoRA](https://github.com/OptimalScale/LMFlow?tab=readme-ov-file#lora), [LISA](https://github.com/OptimalScale/LMFlow?tab=readme-ov-file#lisa), [Flash Attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md), and other acceleration techniques.

## License

Hymba models are released under the [NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).


## Trouble Shooting
Please refer to the [Trouble Shooting](https://github.com/NVlabs/hymba/blob/main/readme/TROUBLESHOOTING.md) page for help.

## Citation

If you find our work helpful, please consider citing our paper:
```
@article{hymba2024,
      title={A Hybrid-head Architecture for Small Language Models}, 
      author={Xin Dong and Yonggan Fu and Shizhe Diao and Wonmin Byeon and Zijia Chen and Ameya Sunil Mahabaleshwarkar and Shih-Yang Liu and Matthijs Van Keirsbilck and Min-Hung Chen and Yoshi Suhara and Yingyan Celine Lin and Jan Kautz and Pavlo Molchanov},
      journal={arXiv preprint arXiv:xxxx},
      year={2024},
      url={https://arxiv.org/abs/xxxx}, 
}
```