# PhoBERT Medical Project

## Author

**Group 6**
* Chu Anh Trường - 23020577 - 33%
* Lê Minh Quân - 23020563 - 33%
* Phạm Quang Vinh- 23020580 - 33%

## Description

The project focuses on building a **Small Language Model (SLM)** for the medical domain in Vietnam, specifically designed to classify medical statements as **True or False**.

The system utilizes **PhoBERT** (a pre-trained language model for Vietnamese) as the core backbone, fine-tuned using a **Discriminative approach**. We employ a **Teacher-Student** methodology where a Large Language Model (e.g., Qwen-32B) acts as a "Teacher" to generate high-quality training data (~360,000 samples) from reputable sources such as **Medlatec**, **Vinmec**, **MedlinePlus**, and **PubMedQA**. The goal is to create a lightweight, high-accuracy model (<1B parameters) suitable for deployment on low-resource devices while ensuring medical accuracy.

**Key Features:**
* **Medical Fact-Checking:** Automatically verifies the accuracy of statements regarding Diseases, Symptoms, and Medications.
* **High-Quality Dataset:** ~360k clean samples generated via Cross-lingual Knowledge Distillation and rigorous cleaning pipelines.
* **Optimized for Vietnamese:** Leverages PhoBERT's robust word segmentation and embedding capabilities for precise understanding of Vietnamese medical terminology.

## Contribute

Pull requests are always welcome. For major changes, please open an issue first to discuss what you want to change before making changes. Any contributions to improve the dataset quality or model architecture are always welcome.

## Project status

This project is **Completed** for the semester requirements.

## Notes

This project is written for **educational and research purposes**. The model's predictions are based on statistical probability and should not replace professional medical advice. Always consult a doctor for health concerns.
