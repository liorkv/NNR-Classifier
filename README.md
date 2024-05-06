# Nearest Neighbor Radius (NNR) Classifier


## Overview

This project contains an implementation of a classifier that uses a nearest neighbor approach with a specified radius. It reads data from CSV files for training, validation, and testing. The classifier is trained and validated to find the optimal radius and then makes predictions on the test data using the optimal radius. Finally, the classification accuracy is calculated and printed.

## Usage

To use the classifier, follow these steps:

1. **Prerequisites**: Ensure you have Python 3 installed on your system and the necessary dependencies listed below.

2. **Clone the repository**:
    ```bash
    git clone [REPOSITORY_URL]
    cd [REPOSITORY_NAME]
    ```

3. **Set up the configuration**: The project includes a `config.json` file that specifies the paths for the training, validation, and test datasets. You may adjust these paths if necessary.

4. **Run the script**: Execute the main script to start the classification process:
    ```bash
    python main.py
    ```

5. **View the results**: The script will output the classification accuracy and the total time taken for the classification process.

## Dependencies

This project depends on the following Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `sklearn`

You can install them using pip:

```bash
pip install numpy pandas scipy scikit-learn
