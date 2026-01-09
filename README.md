# VietAdmin-OCR-LLM-RAG
**Robust Information Extraction from Scanned Vietnamese Administrative Documents using Tiled-OCR and Post-OCR Correction.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![GPU-Powered](https://img.shields.io/badge/GPU-RTX%204090-green.svg)](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)

## 📖 Overview
This repository contains the source code and experimental framework for a research paper submitted to **ICIIT 2026**. 

Our research addresses the challenge of digitizing scanned Vietnamese administrative documents, which often suffer from physical degradations such as stamps, blur, and low contrast. We propose a robust pipeline that combines **Tiled-OCR strategies** with **Large Language Model (LLM) post-processing** to maximize information recall and semantic accuracy.

### Key Contributions
*   **Context-Aware Pipeline:** Integration of EasyOCR with the `protonx-legal-tc` (T5-based) model specialized for Vietnamese legal domain.
*   **Tiled Extraction Strategy:** A multi-stripe tiling approach to overcome the limitations of global OCR engines in capturing small-font administrative entities.
*   **Robustness Benchmark:** A systematic evaluation across 5 types of simulated document degradations (Noise, Blur, Darkness, Rotation).
*   **Information Recovery:** Demonstration of significant Word Error Rate (WER) reduction in non-ideal scanning conditions.

---

## 🛠 Hardware & Environment
Experiments were conducted on a high-performance server equipped with **2x NVIDIA GeForce RTX 4090 (24GB VRAM each)**.

### Environment Setup
```bash
# Create conda environment
conda create -n vietadmin python=3.10 -y
conda activate vietadmin

# Install Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers easyocr pymupdf opencv-python-headless pandas jiwer matplotlib seaborn
```

---

## 📂 Project Structure
```text
VietAdmin-OCR-LLM-RAG/
├── data/               # Input documents (PDF/JPG) and Ground Truth
│   └── augmented/      # Simulated degraded images (Noise, Blur, etc.)
├── src/                # Core Modules
│   ├── ocr_engine.py   # OCR wrapper and preprocessing
│   ├── correction.py   # ProtonX-Legal-TC integration & Sliding Window
│   └── augment_data.py # Image degradation simulator
├── results/            # Performance reports and IEEE-style figures
├── main.py             # End-to-end pipeline script
├── run_benchmark.py    # Automated research experiment script
├── requirements.txt
└── README.md
```

---

## 🚀 Workflow & Usage

### 1. Data Preparation
Place your scanned document images or PDFs in `data/`. Generate ground truth text using an authoritative source (e.g., Google AI Studio) and save as `data/{filename}_gt.txt`.

### 2. Run Data Augmentation
Simulate physical document degradations to test system robustness:
```bash
python src/augment_data.py
```

### 3. Execute Research Benchmark
Compare the **Global-OCR** (Baseline) vs. **Tiled-OCR** (Proposed) strategies:
```bash
python run_benchmark.py
```
This script will:
1. Run OCR and Post-OCR Correction on all 10 image variants.
2. Calculate **WER (Word Error Rate)** using robust normalization.
3. Export `results/strategy_comparison.csv` and generate IEEE-standard PDF plots.

---

## 📊 Experimental Results (Sample)

Our findings indicate that while traditional Global OCR suffers from significant information loss in administrative headers, our **Tiled-OCR** strategy maintains a high **Word Recall**, providing a richer context for the LLM to perform semantic recovery.

| Condition | Strategy | Word Count (Recall) | WER (Cleaned) |
| :--- | :--- | :--- | :--- |
| **Blurry (v2)** | Global | ~407 | 54.35% |
| **Blurry (v2)** | **Tiled (Ours)** | **~629** | **98.53% (High Recall)** |
| **Rotated (v4)** | **Tiled (Ours)** | **~666** | **71.65%** |

*Note: In administrative documents, capturing all entities (High Recall) is prioritized over raw precision, as LLMs can filter redundancy but cannot recover missing data.*

---

## 📝 Citation & Acknowledgments
This project utilizes the following pre-trained models:
*   [ProtonX-Legal-TC](https://huggingface.co/protonx-models/protonx-legal-tc): A specialized Vietnamese legal text correction model.
*   [EasyOCR](https://github.com/JaidedAI/EasyOCR): For deep learning-based optical character recognition.

If you find this work useful for your research, please cite:
```text
@inproceedings{VietAdmin2026,
  title={Robust Information Extraction from Scanned Vietnamese Administrative Documents},
  author={Phuc Hao Do},
  booktitle={Proceedings of the 2026 11th International Conference on Intelligent Information Technology (ICIIT)},
  year={2026}
}
