---
{{ card_data }}
---

# 🖼️ {{ pretty_name | default("Dataset Name", true) }} (Extracted from PyTorch Vision)

{{ dataset_summary | default("", true) }}

## ℹ️ Dataset Details

## 📖 Dataset Description

{{ dataset_description | default("", true) }}

## 📂 Dataset Structure

Each data point is a pair:

- **image:** A visual captured (stored as a PIL Image).
- **label:** The corresponding label (an integer representing the class).

## 🗄️ Source Data

{{ source_data | default("[More Information Needed]", true) }}

## 📜 License

{{ license | default("[More Information Needed]", true) }}