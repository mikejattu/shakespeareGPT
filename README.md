# Shakespeare GPT Text Generator

This project implements a basic GPT-like language model trained on Shakespearean text. The model generates text that emulates the style and language of Shakespeare.

## Project Structure

- **`main.py`**: The main script to train the model and generate text.
- **`GPTCode.py`**: Contains utility functions for text processing and the `BigramLanguageModel` class definition.
- **`selfattention.py`**: Implements transformer model components, including attention heads, multi-head attention, and transformer blocks.

## Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA (for GPU support)

Install the required Python packages using:

pip install torch

Setup
Prepare the Data
Place your Shakespearean text data in a file named input.txt in the same directory as main.py.

Run the Model
Execute the following command to train the model and generate text:
