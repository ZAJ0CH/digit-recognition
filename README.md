# Handwritten Digit Recognizer (MNIST + Custom Data)

This project is a GUI-based handwritten digit recognition app built with PyTorch, Tkinter, and TensorBoard support. It uses a convolutional neural network (CNN) to classify digits drawn by the user or loaded from MNIST/custom datasets.

## Features

* Train a CNN on the MNIST dataset
* Test on both standard and custom test datasets
* Interactive GUI to draw digits and get real-time predictions
* Save misclassified/correct samples
* TensorBoard integration for training visualization
* Model saving and loading

## Requirements

* Python 3.8+
* PyTorch
* torchvision
* Pillow
* Tkinter (usually included with Python)
* TensorBoard

Install dependencies using:

pip install torch torchvision pillow tensorboard

## Directory Structure

* models/: Saved models (automatically created)
* tf\_logs/: TensorBoard logs (auto-created)
* custom\_test\_dataset/: Your custom .png test images (named like uuid\_label.png)
* correct/ and incorrect/: Saved drawn samples

## How to Run

1. Start the App

   python digit\_recognizer.py

2. Use the GUI

   * Draw a digit using your mouse
   * Click:

     * Train model: Train on MNIST dataset
     * Test model: Evaluate on test sets
     * Load model: Load saved model (.pth)
     * Clear: Clear canvas
     * Save as Correct/Incorrect: Save drawn image for later use

3. TensorBoard

   To monitor training:

   tensorboard --logdir tf\_logs

   Then open [http://localhost:6006](http://localhost:6006) in your browser.

## Custom Test Dataset

To test on your own images:

* Place .png files in ./custom\_test\_dataset/
* Format filenames like: anything\_LABEL.png (e.g., abc\_5.png)
* Images should be 28x28 or will be resized

## Experiment Results

* Base 64 neurons	20250517-140935
* Base 64 neurons + weight decay	20250517-165103
* Base 64 neurons + dropout	20250517-164436
* Base 64 neurons + batch norm	20250517-165837
* Base 64 neurons + all	20250517-170542
* Base 128 neurons + all	20250517-171347
* Base 128 neurons + all + augmentation	20250517-172218

## License

MIT – Feel free to use and modify!

