---
title: "Sailor: Open Language Models for South-East Asia"
date: 2024-05-01T12:00:00+08:00
weight: 1
# aliases: ["/first"]
# tags: ["Research"]
# draft: true
# comments: false
# description: "Desc Text."
# disable_share: false
# hide_meta: false
# hide_summary: false # to hide summary in list
# hide_footer: false
math: true
# search_hidden: false # to hide from search page
show_reading_time: true
show_bread_crumbs: true
show_post_nav_links: false # the prev/next after the content
show_code_copy_buttons: true
show_word_count: true
# use_hugo_toc: true
# show_toc: true
# toc_open: true # default expand all
# cover:
#     image: "path"
#     # can also paste direct link from external site
#     # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
#     alt: "<alt text>"
#     caption: "<text>"
#     relative: true # To use relative path for cover image, used in hugo Page-bundles
#     responsive_images: true
#     hidden: false
# header:
#   background: "" # background css value
#   background_image: ""
#   gradient: false
#   blur: false
---
{{< button href="https://arxiv.org/abs/2404.03608" label="PAPER" external=true >}}
{{< button href="https://github.com/sail-sg/sailor-llm" label="GITHUB" external=true >}}
{{< button href="https://huggingface.co/collections/sail/sailor-65e19a749f978976f1959825" label="HUGGING FACE" external=true >}}
{{< button href="https://huggingface.co/spaces/sail/Sailor-14B-Chat" label="DEMO" external=true >}}


## Introduction

Sailor is a suite of Open Language Models tailored for South-East Asia (SEA), focusing on languages such as 🇮🇩Indonesian, 🇹🇭Thai, 🇻🇳Vietnamese, 🇲🇾Malay, and 🇱🇦Lao. Developed with careful data curation, Sailor models are designed to understand and generate text across the diverse linguistic landscapes of the SEA region. Built from Qwen 1.5, Sailor encompasses models of varying sizes, spanning from 0.5B to 14B versions for different requirements.

Key features of Sailor include:

- Continually pretrained on **200 Billion to 400 Billion** tokens over 7 languages, including Indonesian, Thai, Vietnamese, Malay, Lao, English, and Chinese.
- Various model sizes (**0.5B**, **1.8B**, **4B**, **7B**, **14B**) to support different requirements.
- Strong performance on SEA benchmarks such as XQuAD, TydiQA, XCOPA, Belebele, and M3Exam.
- No restrictions on research and commercial use, but must comply with the Qwen 1.5 license.

## Built from Open-Source Community

Sailor owes its existence to the open-source community. It is crafted by continually pre-training from language models like the remarkable [Qwen 1.5](https://qwenlm.github.io/blog/qwen1.5/) models, which already have great performance on SEA languages. The pre-training corpus heavily leverages the publicly available corpus, including [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B), [SkyPile](https://huggingface.co/datasets/Skywork/SkyPile-150B), [CC100](https://huggingface.co/datasets/cc100), and [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400).

By employing aggressive data deduplication and careful data cleaning on the collected corpus, we have attained a high-quality dataset spanning various languages. Through systematic experiments to determine the weights of different languages, Sailor models undergo training from 200B to 400B tokens, tailored to different model sizes. The approach boosts their performance on SEA languages while maintaining proficiency in English and Chinese without significant compromise. Finally, we continually pre-train the Qwen1.5-0.5B model with 400 Billion tokens, and other models with 200 Billion tokens to obtain the Sailor models.

For most of the models, we use 200 billion tokens, with the effective tokens for each language as shown below. For models utilizing 400 billion tokens, they are doubled accordingly.

| **Language**         | **Tokens (Billion)** |
|----------------------|----------------------|
| Indonesian (id)      | 51.56                |
| Malay (ms)           | 7.91                 |
| Thai (th)            | 38.24                |
| Vietnamese (vi)      | 41.50                |
| Lao (lo)             | 0.34                 |
| English (en)         | 37.2                 |
| Chinese (zh)         | 22.64                |

## Commitment to Open-Source Community

The release of Sailor models marks the beginning of our commitment to open-source. In the coming weeks, we plan to release:
- Training recipes
- Code for pre-training
- Pipeline for data cleaning and deduplication
- Pre-training corpus

## Benchmarking Performance

Sailor models are evaluated across several high-quality benchmarks, encompassing four kinds of different tasks: question answering, commonsense reasoning, reading comprehension, and examination. We gratefully acknowledge the contributions of all dataset authors. For evaluation, following established protocols, we employed the awesome evaluation platform [OpenCompass](https://github.com/open-compass/opencompass) for comprehensive evaluation. The performance of all models is assessed based on the 3-shot Exact Match performance, with prompts provided in local languages (e.g., Indonesian task description for Indonesian tasks).

We acknowledge and respect the release of several SEA language models before, including [SEA-LION](https://aisingapore.org/aiproducts/sea-lion/), [SeaLLMs](https://arxiv.org/abs/2312.00738), [Typhoon](https://arxiv.org/abs/2312.13951), and [VinaLLaMA](https://arxiv.org/abs/2312.11011). Here we mainly selected [SeaLLMs-7B-Hybrid](https://huggingface.co/SeaLLMs/SeaLLM-7B-Hybrid), its base model [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7B), [SeaLLMs-7B-v2](https://huggingface.co/SeaLLMs/SeaLLM-7B-v2), and its base model [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) for performance comparison. Evaluation results for more models will be presented in our paper. Our reporting strictly adheres to the same evaluation methodology to ensure fair comparison, and we make much effort to closely match the reported results of the baseline.

### Evaluation Tasks

- **Question Answering**: [XQuAD](https://arxiv.org/abs/1910.11856) (Thai, Vietnamese) and [TydiQA](https://arxiv.org/abs/2003.05002) (Indonesian).
- **Commonsense Reasoning**: [XCOPA](https://aclanthology.org/2020.emnlp-main.185/) (Indonesian, Thai, Vietnamese).
- **Reading Comprehension**: [Belebele](https://arxiv.org/abs/2308.16884) (Indonesian, Thai, Vietnamese).
- **Examination**: [M3Exam](https://arxiv.org/abs/2306.05179) (Javanese, Thai, Vietnamese).

### Question Answering

All models are evaluated on the XQuAD and TydiQA benchmarks, with the 3-shot Exact Match (EM) and F1 score reported. Baselines which have better performance than Sailor models are highlighted in green. Sailor-14B's XQuAD (th) result seem abnormal, with its predictions tending to be semantically equivalent but longer than the groundtruth, leading to lower EM and F1 scores compared to Qwen1.5.

| 3-shot (EM / F1) | XQuAD (th) | TydiQA (id) | XQuAD (vi) |
|-----------------|------------|-------------|------------|
| Qwen1.5-0.5B | 14.19 / 23.35 | 20.71 / 32.64 | 19.85 / 35.38 |
| **Sailor-0.5B** | 15.84 / 27.58 | 30.44 / 54.74 | 21.13 / 40.57 |
| Qwen1.5-1.8B | 27.24 / 43.56 | 29.73 / 53.76 | 29.17 / 48.15 |
| **Sailor-1.8B** | 32.72 / 48.66 | 40.88 / 65.37 | 34.22 / 53.35 |
| Qwen1.5-4B | 34.03 / 53.40 | 48.32 / 72.68 | 43.71 / 63.86 |
| **Sailor-4B** | 46.82 / 63.34 | 53.98 / 73.48 | 47.65 / 67.09 |
| Llama-2-7B | 30.64 / 43.80 | 56.64 / 72.14 | 46.96 / 66.16 |
| Mistral-7B-v0.1 | 48.48 / 63.27 | 63.54 / 78.73 | 53.72 / 72.75 |
| SeaLLM-7B-Hybrid | 49.70 / 67.62 | 50.62 / 75.21 | 49.62 / 70.74 |
| SeaLLM-7B-v2 | 34.55 / 55.13 | 52.21 / 77.00 | 46.19 / 72.11 |
| Qwen1.5-7B | 53.79 / 69.30 | 57.17 / 77.28 | 56.63 / 76.99 |
| **Sailor-7B** | 57.88 / 71.06 | 60.53 / 75.42 | 53.81 / 74.62 |
| Qwen1.5-14B | 55.53 / 74.36 | 60.18 / 81.05 | 57.57 / 77.58 |
| **Sailor-14B** | 49.43* / 69.99* | 58.94 / 77.85 | 57.83 / 77.37 |

### Commonsense Reasoning

All models are evaluated on the XCOPA benchmark, with the 3-shot accuracy reported.

| 3-shot (EM) | XCOPA (th) | XCOPA (id) | XCOPA (vi) |
|-------------|------------|------------|------------|
| Random Guess | 50.00 | 50.00 | 50.00 |
| Qwen1.5-0.5B | 51.00 | 52.20 | 53.80 |
| **Sailor-0.5B** | 51.00 | 58.20 | 58.00 |
| Qwen1.5-1.8B | 52.60 | 51.60 | 53.40 |
| **Sailor-1.8B** | 53.80 | 64.20 | 63.20 |
| Qwen1.5-4B | 53.40 | 55.00 | 57.80 |
| **Sailor-4B** | 53.40 | 69.20 | 68.20 |
| Llama-2-7B | 52.80 | 64.00 | 62.00 |
| Mistral-7B-v0.1 | 57.20 | 62.40 | 61.60 |
| SeaLLM-7B-Hybrid | 58.20 | 71.60 | 67.60 |
| SeaLLM-7B-v2 | 56.80 | 64.00 | 64.60 |
| Qwen1.5-7B | 54.20 | 62.20 | 66.20 |
| **Sailor-7B** | 59.00 | 72.20 | 72.20 |
| Qwen1.5-14B | 60.00 | 72.20 | 74.00 |
| **Sailor-14B** | 64.40 | 79.60 | 80.40 |

### Reading Comprehension

All models are evaluated on the Belebele benchmark, with the 3-shot Exact Match (EM) reported. Baselines which have better performance than Sailor models are highlighted in green.

| 3-shot (EM) | Belebele (th) | Belebele (id) | Belebele (vi) |
|-------------|---------------|---------------|---------------|
| Random Guess | 25.00 | 25.00 | 25.00 |
| Qwen1.5-0.5B | 29.89 | 26.89 | 30.22 |
| **Sailor-0.5B** | 32.22 | 30.89 | 32.33 |
| Qwen1.5-1.8B | 30.11 | 32.00 | 31.33 |
| **Sailor-1.8B** | 34.22 | 34.89 | 35.33 |
| Qwen1.5-4B | 32.78 | 36.22 | 35.22 |
| **Sailor-4B** | 36.11 | 41.33 | 38.89 |
| Llama-2-7B | 31.78 | 39.78 | 38.00 |
| Mistral-7B-v0.1 | 34.33 | 41.33 | 41.33 |
| SeaLLM-7B-Hybrid | 37.78 | 43.11 | 43.00 |
| SeaLLM-7B-v2 | 36.33 | 43.11 | 47.00 |
| Qwen1.5-7B | 38.33 | 42.00 | 42.89 |
| **Sailor-7B** | 41.56 | 44.33 | 45.33 |
| Qwen1.5-14B | 41.44 | 46.22 | 40.33 |
| **Sailor-14B** | 42.11 | 47.56 | 45.89 |

### Examination

All models are evaluated on the M3Exam benchmark, with the 3-shot Exact Match (EM) reported. The code jv is short for Javanese, which is a language spoken in Indonesia.

| 3-shot (EM) | M3Exam (th) | M3Exam (jv) | M3Exam (vi) |
|-------------|-------------|-------------|-------------|
| Qwen1.5-0.5B | 22.38 | 22.10 | 29.12 |
| **Sailor-0.5B** | 21.87 | 28.84 | 23.53 |
| Qwen1.5-1.8B | 23.81 | 26.15 | 36.39 |
| **Sailor-1.8B** | 23.90 | 29.65 | 27.67 |
| Qwen1.5-4B | 26.26 | 30.19 | 40.02 |
| **Sailor-4B** | 27.23 | 29.11 | 31.58 |
| Llama-2-7B | 21.13 | 23.99 | 34.14 |
| Mistral-7B-v0.1 | 29.59 | 31.00 | 43.54 |
| Sea-Lion-7B | 23.90 | 21.56 | 26.89 |
| SeaLLM-7B-Hybrid | 25.98 | 24.53 | 38.79 |
| SeaLLM-7B-v2 | 35.60 | 29.92 | 50.36 |
| Qwen1.5-7B | 35.88 | 33.15 | 51.09 |
| **Sailor-7B** | 38.33 | 35.85 | 51.98 |
| Qwen1.5-14B | 43.18 | 35.04 | 58.47 |
| **Sailor-14B** | 48.22 | 39.89 | 60.54 |


## Contributors

- Longxu Dou, Sea AI Lab
- Qian Liu, Sea AI Lab
- Guangtao Zeng, SUTD
- Jia Guo, NUS
- Jiahui Zhou, Sea AI Lab
- Ziqi Jin, SUTD
- Xin Mao, NTU
- Wei Lu, SUTD
- Min Lin, Sea AI Lab

## Contact Us

Sailor models are free for research and commercial use, but you should also obey the [Qwen 1.5 license](https://huggingface.co/Qwen/Qwen1.5-0.5B/blob/main/LICENSE). We encourage you to use Sailor models in your research and applications.

For questions or collaboration, please:
- Raise an issue in our GitHub repository
- Contact us at:
  - [doulx@sea.com](mailto:doulx@sea.com)
  - [liuqian.sea@gmail.com](mailto:liuqian.sea@gmail.com)

## Citation

```
@article{dou2024sailor,
  title={Sailor: Open Language Models for South-East Asia},
  author={Dou, Longxu and Liu, Qian and Zeng, Guangtao and Guo, Jia and Zhou, Jiahui and Lu, Wei and Lin, Min},
  journal={arXiv preprint arXiv:2404.03608},
  year={2024}
}
```