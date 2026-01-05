# Medical U-Net Segmentation

This repository contains a small medical image segmentation project using a U-Net architecture.

## Overview
The goal of this project is to perform binary medical image segmentation using a convolutional neural network.
The model is trained and evaluated using the Dice coefficient, which is commonly used in medical image analysis.

## Model
- Architecture: U-Net
- Loss function: Weighted Binary Cross-Entropy + Dice loss
- Evaluation metric: Dice coefficient

## Dataset
The project uses preprocessed medical image data stored in `.npz` format.
Each sample consists of:
- an input image
- a corresponding binary segmentation mask

## Project Structure
