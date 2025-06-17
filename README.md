<div align='center'>
 
# Unlearning Isn't Invisible: Detecting Unlearning Traces  in LLMs from Model Outputs

<!-- [![preprint](https://img.shields.io/badge/arXiv-2506.04205-B31B1B)](https://arxiv.org/abs/2506.04205) -->

<!-- [![Venue:NeurIPS 2024](https://img.shields.io/badge/Venue-NeurIPS%202024-blue)](https://neurips.cc/Conferences/2024) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/Unlearn-Trace?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/Unlearn-Trace)](https://github.com/OPTML-Group/Unlearn-Trace)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/Unlearn-Trace)](https://github.com/OPTML-Group/Unlearn-Trace)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/Unlearn-Trace)](https://github.com/OPTML-Group/Unlearn-Trace)

</div>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="./images/teasor.png" alt="teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 11px;">  <strong style="font-size: 11px;">Figure 1:</strong> Schematic overview of unlearning trace detection.</em>
    </td>
  </tr>
</table>

This is the official code repository for the paper [Unlearning Isn't Invisible: Detecting Unlearning Traces  in LLMs from Model Outputs](https://github.com/OPTML-Group/Unlearn-Trace).


## Release 

- We will release our code base very soon!
- [6/16] We have uploaded our paper to the Arxiv platform!

## WMDP Unlearning

RMU unlearning

## Installation

We build up our code repo based on [this URL](https://github.com/locuslab/llm-idiosyncrasies).

Please use separate environments for response generation and classification.

1. Setup for Response Generation
```
conda create -n generation python=3.9 -y
conda activate generation
pip install vllm==0.6.3.post1 datasets==3.2.0 openai 
```

2. Setup for Classification
```
conda create -n classification python=3.9 -y
conda activate classification
pip install llm2vec==0.2.3 tensorboard
```

## Response Generation

Forget-relevant response generation

Forget-irrelevant response generation

## Contributors
* [Yiwei Chen](https://yiwei-chenn.github.io/)
* [Soumyadeep Pal](https://scholar.google.ca/citations?user=c2VU-_4AAAAJ&hl=en)

## Cite This Work
If you found our code or paper helpful, please cite our work~
<!-- ```
@article{chen2025safety,
  title={Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-tuning},
  author={Chen, Yiwei and Yao, Yuguang and Zhang, Yihua and Shen, Bingquan and Liu, Gaowen and Liu, Sijia},
  journal={arXiv preprint arXiv:2503.11832},
  year={2025}
}
``` -->