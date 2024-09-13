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
```bash
pip install torch
```

## Setup
### Prepare the Data
Place your Shakespearean text data in a file named **`input.txt`** in the same directory as **`main.py`**.

### Run the Model
Execute the following command to train the model and generate text:
```bash
python main.py
```
This will:

1. Load and preprocess the text data.
2. Train a GPT-like model using the provided data.
3. Generate text based on the trained model.
4. Save the generated text to output.txt.

## Model Configuration

The model is configured with the following hyperparameters:

#### **Batch Size**: 64
#### **Block Size**: 256
#### **Iterations**: 5000
#### **Learning Rate**: 3e-4
#### **Embedding Size**: 384
#### **Number of Heads**: 6
#### **Number of Layers**: 6
#### **Dropout Rate**: 0.2

## How It Works
### Data Processing
  The text data is loaded and encoded into integer sequences.
  The data is split into training and validation sets.

### Model Training
  The BigramLanguageModel class defines a GPT-like model with positional encoding and multi-head attention.
  The model is trained on the encoded text data for a specified number of iterations.

### Text Generation
  After training, the model generates new text sequences based on learned patterns from the Shakespearean data.

## Usage

  **Text Generation**: The model generates text based on the trained data. Adjust the max_new_tokens parameter in the generate function to control the length of the generated text.

## Acknowledgements
Inspired by the GPT architecture and transformer models.
This project was inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) implementation by [Andrej Karpathy](https://github.com/karpathy). His work provided valuable insights and guidance for building this GPT-like model.
