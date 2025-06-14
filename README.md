# Obliviate: Efficient Unmemorization for LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2502.15010-b31b1b.svg)](https://arxiv.org/abs/2502.15010)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of "Obliviate: Efficient Unmemorization for Protecting Intellectual Property in Large Language Models".

Recent copyright agreements between AI companies and content creators have highlighted the need for precise control over language models' ability to reproduce copyrighted content. While existing approaches rely on either complete concept removal through unlearning or simple output filtering, we propose Obliviate, a novel post-training technique that selectively prevents verbatim reproduction of specific text while preserving semantic understanding.
Obliviate operates by selecting tokens within memorized sequences and modifying the model's probability distribution to prevent exact reproduction while maintaining contextual understanding. We evaluate Obliviate on multiple large language models (LLaMA-3.1 8B, LLaMA-3.1-instruct 8B, Qwen-2.5-7B, and Yi-1.5 6B) across both synthetic memorization tasks and organic copyright content. Our results demonstrate that Obliviate achieves orders of magnitude reduction, e.g., 100x, in verbatim memorization while maintaining model performance within 1% of baseline on standard benchmarks (HellaSwag, MMLU, TruthfulQA, and Winogrande). This makes Obliviate particularly suitable for practical deployment scenarios where companies need to efficiently address copyright concerns in pretrained models without compromising their general capabilities.

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

The project is organized as follows:
- **config**: 
  - supported model finetuning parameters 
  - obliviate prefix and stride parameters e.g. 10-5-1.json:
    ```json
        {
        "start": 10,
        "stride": 5,
        "span": 1
        }
    ```
    *start*:  start unmemorizing 10 tokens into target text  
    *stride*: skip 5 tokens betweeen unmemorize tokens  
    *span*:   unmemorize 1 token at each stride
- **data**: datasets for synthetic and organic targets
- **experiments**: configuration for model memorization and obliviate runs
- **src**: shell scripts and python files for memorizing and unmemorization

## Running Experiments
To unmemorize a target, first create an experiment configuration. This example shows the properties to specify: 
```json
{
    "base_directory": "/datadrive2/unmemorize/experiments/3",
    "experiments": [
      {
        "model_name": "llama3.1-8b",
        "configurations": [
          {
            "config": "10-5-1",
            "sample_count": 1,
            "experiment_types": [
              {
                "data": "synthetic",
                "name": "standard",
                "top_k": 5,
                "smart_select": false
              },
              {
                "data": "pretrain",
                "name": "standard",
                "top_k": 5,
                "smart_select": false
              }                            
            ]
          }
        ]
      }
    ]
}
```
- *base_directory* specifies the directory into which experiment outputs will be placed
- *config* must match a configuration in the config/runs directory. 
- *sample_count* is the number of samples from the specified dataset to unmemorize 
- *data* must either be *pretrain* for the organic target dataset, *synthetic* or *synthetic100* (note that the organic dataset does not include text from Harry Potter)
- *name* is either *standard* or *smart* for the token selection algorithm
- *top_k* is the number of tokens to preserve for k/l loss
- *smart_select* specifies if unmemorize token selection...

### Memorization
If the experiment targets synthetic data, first have the model memorize the data:

```bash
cd src
./memorize.sh ../experiments/config/<model configuration>
```
The memorization places the memorized model **memorized** directory under the model name in *base_directory*. 

### Unmemorization
To unmemorize, run the unmemorize script with the experiment to execute. If the experiment configuration specifies a synthetic dataset, you must run the memorize step first. 
```bash
cd src
./unmemorize.sh ../experiments/<experiment>/<model configuration> 
```
For, example, to run experiment 2 for llama3.1-8b:
```bash
cd src
./unmemorize.sh ../experiments/2/llama3.1-8b.json 
```
The script stores unmemorized models, benchmark results and test logs under <base_directory>/{smart/standard}/{synthetic/pretrain}/{model name]/{run config}/0

The example above would put the 10-0-1 run here: 
```bash
/datadrive2/unmemorize/experiments/2/standard/synthetic/10-5-1/0
```
### Result Plots
Experiment result plots for longest common sequence, bleu and rouge2 metrics, and benchmarks are placed in the base directory.



 
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
