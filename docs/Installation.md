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