# LLM Hallucination Detection

This project provides a framework for detecting hallucinations in Large Language Models (LLMs). It includes various detection methods and supports different tasks like Multiple Choice Question Answering (MCQA), Question Answering (QA), and Truthful QA.

## Project Structure

```
├── .gitignore
├── scripts/
│   ├── run-mcqa.sh
│   ├── run-qa.sh
│   └── run-truthful_qa.sh
└── src/
    ├── detect_methods/
    │   ├── __init__.py
    │   ├── activation_decoding_entropy.py
    │   ├── eigen_score.py
    │   ├── llm_check.py
    │   ├── perplexity.py
    │   ├── self_check.py
    │   └── verbalized.py
    ├── fine-tune/
    │   ├── Info-Llama-31-8B.yaml
    │   ├── Info-Qwen-7B.yaml
    │   ├── Truth-Llama-31-8B.yaml
    │   ├── Truth-Qwen-7B.yaml
    │   ├── dataset_info.json
    │   ├── ds_z3_config.json
    │   └── preprocess.py
    ├── requirements.txt
    ├── task/
    │   ├── __init__.py
    │   ├── _detection.py
    │   ├── _generation.py
    │   ├── _qa_utils.py
    │   ├── mcqa.py
    │   ├── qa.py
    │   └── truthful_qa.py
    └── utils.py
```

- **scripts/**: Contains shell scripts to run the different tasks.
- **src/detect_methods/**: Implements various hallucination detection methods.
- **src/fine-tune/**: Contains configurations and scripts for fine-tuning models.
- **src/task/**: Contains the implementation of the main tasks (MCQA, QA, Truthful QA).
- **src/requirements.txt**: Lists the Python dependencies for this project.
- **src/utils.py**: Contains utility functions.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd cot-hallu-detect
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r src/requirements.txt
    ```

## Running the Tasks

The `scripts` directory provides shell scripts to easily run the different tasks. You will need to modify the scripts to provide the correct paths to your models and datasets.

### Multiple Choice Question Answering (MCQA)

To run the MCQA task, use the `run-mcqa.sh` script:

```bash
bash scripts/run-mcqa.sh
```

Make sure to update the paths in `scripts/run-mcqa.sh` for the model (`-m`) and dataset (`-d`).

### Question Answering (QA)

To run the QA task, use the `run-qa.sh` script:

```bash
bash scripts/run-qa.sh
```

Update the paths in `scripts/run-qa.sh` for the model (`-m`), NLI model (`--nli`), embedding model (`--embd`), and dataset (`-d`).

### Truthful Question Answering (Truthful QA)

To run the Truthful QA task, use the `run-truthful_qa.sh` script:

```bash
bash scripts/run-truthful_qa.sh
```

Update the paths in `scripts/run-truthful_qa.sh` for the model (`-m`), NLI model (`--nli`), embedding model (`--embd`), dataset (`-d`), truth model (`--truth`), and info model (`--info`).

## Detection Methods

This project implements several hallucination detection methods, including:

-   **Activation Decoding Entropy**: Measures the entropy of the model's activations.
    Shiqi Chen, Miao Xiong., et al, "In-Context Sharpness as Alerts: An Inner Representation Perspective for Hallucination Mitigation," in ICLR 2024 Workshop on Reliable and Responsible Foundation Models, 2024.
-   **Eigen Score**: Uses the eigenvalues of the model's attention mechanism.
    Chao Chen,  Kai Liu., et al, "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection," in The Twelfth International Conference on Learning Representations, 2024.
-   **LLM Check**: Uses another LLM to check the validity of the generated text.
    Gaurang Sriramanan, Siddhant Bharti., et al, "LLM-Check: Investigating Detection of Hallucinations in Large Language Models," in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024, pp. 915–932.
-   **Perplexity**: Measures the perplexity of the generated text.
-   **Self Check**: The model checks its own generated text for consistency.
    Potsawee Manakul, Adian Liusie., et al, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," in The 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 9004–9017.
-   **Verbalized**: Based on verbalized confidence.
    Kumar, A., et al, "Confidence Under the Hood: An Investigation into the Confidence-Probability Alignment in Large Language Models," in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 315-334.

## Fine-tuning

The `src/fine-tune` directory contains configurations for fine-tuning models like Llama and Qwen for truthfulness and informativeness. The `preprocess.py` script can be used to prepare your dataset for fine-tuning.