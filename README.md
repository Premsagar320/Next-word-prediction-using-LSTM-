# Next Word Prediction using LSTM (From Scratch)

This project implements a character-level LSTM model completely from scratch in Python. Instead of using deep learning frameworks for the LSTM internals, all the gates, activations, forward pass, backward pass, and parameter updates were coded manually. The goal of this project was to understand how recurrent networks work at a low level, especially how gradients move through time.

## What the Project Does

The model is trained on text from *Sherlock Holmes* to learn character-level patterns. Given a sequence of characters, the model predicts the next character. Over time, it learns to generate Sherlock-style text.

## What Was Implemented Manually

- Input, forget, and output gates  
- Cell state and hidden state updates  
- Backpropagation Through Time (BPTT)  
- A lightweight custom Adam optimizer  
- Sampling method for text generation  

No built-in LSTM modules were used — everything is coded using basic NumPy operations.

## Why This Project

The main purpose was to get hands-on understanding of:
- How LSTM gates work internally  
- How gradients flow across many time steps  
- How optimizers update recurrent parameters  
- How sequence models learn structure in text  

This project acts as a learning experiment rather than a production model.



