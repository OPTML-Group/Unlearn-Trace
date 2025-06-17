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

We will release our code base very soon~
<!-- - [3/14] We have uploaded our first version of [Safety Mirage](https://arxiv.org/abs/2503.11832) to the Arxiv platform. -->

<!-- ## Install

Our safety-unlearn framework has been developed on the LLaVA-1.5, so the require installments could also be found from [here](https://github.com/haotian-liu/LLaVA).
Also, you could use following steps:

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/OPTML-Group/VLM-Safety-MU
cd VLM-Safety-MU
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Unlearning Fine-tune
Our base model LLava-1.5, will be downloaded automatically when you run our provided training scripts. No action is needed.

For full-parameter unlearning fine-tune, you should run
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/finetune_unlearn.sh
```

For LoRA unlearning fine-tune, you should run
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/finetune_unlearn_lora.sh
```

Some unlearn options to note:

- `--unlearn_type`: unlearning algorithm type, which could be 'npo' or 'rmu'.
- `--rmu_XXX`: are the specific hyperparameters for rmu algortihm.
- `--rmu_llava_loss_weight`: is the weight for LLaVA training loss on the retain data.
- `--rmu_retain_alpha`: is the weight for rmu loss on the retain data.
- `--npo_beta`: is the balancing parameter for npo algortihm.
- `--npo_forget_alpha`: is the weight for npo loss on the forget data.
- `--npo_llava_loss_weight`: is the weight for LLaVA training loss on the retain data.

Also, the data path and the output dictionary should also be specified~ -->

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