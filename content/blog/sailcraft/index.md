---
title: "SailCraft: Data Toolkit for Sailor Language Models"
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
{{< button href="https://github.com/sail-sg/sailcraft" label="GITHUB" external=true >}}

## Introduction

Today, we are thrilled to introduce the **SailCraft** data processing pipeline, a comprehensive open-source solution for large language model dataset curation. Built with meticulous attention to detail, Sailcraft represents a significant advancement in data preprocessing techniques for machine learning models.

The pipeline encompasses a sophisticated four-stage data cleaning approach:
1. Initial data cleaning
2. Near deduplication
3. Exact deduplication
4. Second-round data cleaning

With a particular emphasis on linguistic diversity, Sailcraft provides specialized cleaning capabilities for a wide range of languages, including Arabic, Bengali, Catalan, Spanish, Basque, French, Hindi, Portuguese, Urdu, and optimized processing for English, Indonesian, Vietnamese, Chinese, Thai, Lao, and Malay.

## Key Capabilities

Sailcraft distinguishes itself through its robust and flexible data processing framework. Researchers and developers can leverage this tool to:
- Obtain granular filtered data counts at each processing stage
- Implement language-specific cleaning rules with unprecedented ease
- Conduct detailed investigations into data removal processes

The pipeline's design reflects our commitment to transparency and open scientific research. By providing a comprehensive, adaptable data processing solution, we aim to empower the machine learning community with high-quality dataset curation tools.

## Acknowledgements

This project stands as a testament to the collaborative spirit of the open-source community, drawing inspiration and leveraging contributions from critical projects including:
- BigScience data cleaning tool
- Chenghao Mou's all-in-one deduplication tool
- Google's deduplication project

## Looking Forward

We invite the community to explore, utilize, and provide feedback on Sailcraft. Your insights will be crucial in refining and expanding this data processing framework.

## Citation

```
@article{dou2024sailor,
  title={Sailor: Open Language Models for South-East Asia},
  author={Dou, Longxu and Liu, Qian and Zeng, Guangtao and Guo, Jia and Zhou, Jiahui and Lu, Wei and Lin, Min},
  journal={arXiv preprint arXiv:2404.03608},
  year={2024}
}
```