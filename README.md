# Flower Classification App

This is a Streamlit-based web application that classifies flowers from uploaded images using a Deep Learning model.

## Features

*   **Image Upload**: Supports uploading images in JPG, JPEG, and PNG formats.
*   **Real-time Classification**: Uses a pre-trained PyTorch model to classify flowers instantly.
*   **Top 3 Predictions**: Displays the top 3 most likely flower species with their confidence scores.
*   **User-Friendly Interface**: Simple and intuitive interface built with Streamlit.

## Prerequisites

*   Python 3.11.9

## Installation

1.  Clone this repository or download the source code.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2.  Open your web browser and navigate to the URL shown in the terminal.
3.  Upload an image of a flower to see the classification results.

## Project Structure

*   `app.py`: The main Streamlit application script.
*   `src/`: Contains source code for inference and model handling.
*   `configs/`: Contains configuration files, such as class names.
*   `notebooks/`: Jupyter notebooks used for model training and analysis.
*   `requirements.txt`: List of Python dependencies.

## What would I do differently for a production deployment?
* Serving Infrastructure: Wrap the model in a high-performance framework like FastAPI to expose it as a REST API, enabling integration with mobile and web applications.
* Model Optimization: Convert the PyTorch model to ONNX format and apply Int8 Quantization. This would significantly reduce latency and memory usage compared to the current FP32 implementation. But this depends on the devices and the number of user, if the devices is very old and large scale of user, optimization is needed, if not, then it is fine.
* Containerization: Package the application using Docker to ensure consistent environments across development and production, deploying via orchestration tools like Kubernetes for auto-scaling.
* Modify the dataset by adding more train data to ensure the model perform better in all the classes, reduce overfiting and improve the model's generalization
* Add segmentation model the exclude the background of the images and include only flower.
* Dropout could be added to lessen overfitting (unable to test, due to time constraint)
* the ui could also be further improved.
* Chatbot could also be implemented to explain the characteristic of each of the flower, or even explain the model itself, to clarify them to the users.


