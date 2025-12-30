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
