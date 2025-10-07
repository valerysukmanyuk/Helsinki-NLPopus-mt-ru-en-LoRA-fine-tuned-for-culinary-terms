# LoRA fine-tuning of `Helsinki-NLP/opus-mt-ru-en` for culinary terms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)]()

**Summary:** This repository contains experiments with adapting the [**Helsinki-NLP/opus-mt-ru-en**](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en) translation model using Low-Rank Adaptation (LoRA) for culinary terminology. It includes a training notebook, datasets (2 augmented versions), model files, reproducibility instructions, and inference examples.

---

## Table of Contents

1. [Concept](#concept)
2. [Dataset](#dataset)
3. [Augmentation & Reproducibility](#augmentation--reproducibility)
4. [Inference](#inference)
5. [Evaluation](#evaluation)
6. [License](#License)

---

## Concept

The goal is to improve the translation of culinary terms and dish names, preserving the natural restaurant menu style, while minimally modifying the base model. LoRA enables efficient fine-tuning with reduced memory and compute requirements.

## Dataset


| Dataset | Size | Description |
|---------|------|-------------|
| Original dataset | 312 | Menu items from Russian restaurants, manually translated into English |
| Augmented dataset v1 | 907 | Expanded version created via data augmentation |
| Augmented dataset v2 | 1,262 | Further extended dataset with additional augmented entries |

## Augmentation & Reproducibility

Augmented datasets were generated with **GPT-4o-mini** using few-shot prompts, manual moderation and duplicate checks. 
**Example prompt template**

```
f"Here are examples of menu items in Russian with English translations:\n"
f"{examples}\n"
f"Combine two menu items in the same format (Russian - English). They must be similar to the example. Use only words from the examples! No words not from the example are allowed; this is a mandatory requirement!"
```

## Inference

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Loading model
model_name = "my_model_ver2"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Translation function
def translate(text, model=model, tokenizer=tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=max_length)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Example of inference
russian_text = "Бефстроганов с картофельным пюре"
english_translation = translate(russian_text)
print(english_translation)

```

## Evaluation

Metrics used in this experiment: 

* BLEU
* chrF
* ROUGE-L
* human judgment (fluency, adequacy)

Conclusions and metrics are shown in the notebook.

## License

* This repository is MIT licensed.

---
