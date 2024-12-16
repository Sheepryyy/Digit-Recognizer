# Digit-Recognizer
Digit Recognizer project for DNSC 6301

##  1.Basic Information
### Person developing model: Rui Yang, ruiy@gwmail.gwu.edu
### Model date: December, 2024
### Model version: 1.0
### License: MIT
### Model implementation code: digit recognizer.ipynb
### Intended Use
#### Intended uses:
This model is designed to recognize handwritten digits using the MNIST dataset. The primary goal is to train and predict digit labels based on input pixel values. It can be used in the following scenarios:
Educational use: For learning machine learning concepts, including classification tasks.
Basic digit recognition: Small-scale applications requiring offline digit classification.

#### Intended users:
Students and beginners in data science, for hands-on learning of machine learning workflows.
Educators teaching basic machine learning classification problems.

#### Out-of-scope uses:
This model is not suitable for real-time digit recognition applications.
It is not intended for deployment in critical systems requiring high accuracy, such as financial document processing.
This model does not support noisy, low-quality, or non-standard handwritten digit datasets.

## 2. Training Data
### Source of training data:
The training data comes from Kaggle's Digit Recognizer competition page, using the MNIST dataset. The data can be downloaded from the competition's official website, including training data (`train.csv`) and test data (`test.csv`).

### How training data was divided into training and validation data:
In the downloaded dataset, train.csv contains data for training, and test.csv is used for final model prediction. The division of training and validation data is as follows:
Training data: extracted from train.csv and used for training the model.
Validation data: divided into training and validation sets in an 80-20 ratio.

### Number of rows in training and validation data:
Training data (train.csv): 42,000 rows, 785 columns (1 column for labels and 784 columns for pixel features)
Test data (test.csv): 28,000 rows, 784 columns (only contains pixel features)

### Data dictionary:
Below is the description of each column in train.csv:
| **Name**          | **Modeling role**                   | **Measurement level** | **Description**                                              |
|--------------------|-------------------------------------|------------------------|--------------------------------------------------------------|
| `label`           | Target (CNN, SVM, Logistic Regression) | Categorical            | The actual numerical label of the image (0 to 9, 10 categories in total)                          |
| `pixel0` - `pixel783` | Features (CNN, SVM, Logistic Regression) | Continuous             | The grayscale value of each pixel ranges from 0 to 255 (the image is expanded to 28x28 pixels)      |


### Additional Notes:
- Each image consists of 28x28 grayscale pixels, expanded into 784 feature columns (`pixel0` to `pixel783`).  
- Label column: Target variable used in supervised learning tasks.  
- Pixel columns: Input features used for model training and prediction.  

## 3. Testing Data
### Source of testing data:
The testing data comes from Kaggle's Digit Recognizer competition page. You can download the test.csv file through the official website.

### Number of rows in testing data:
The testing data (test.csv) contains 28,000 rows and 784 columns (each column corresponds to a pixel value in the image).

### Differences between training and testing data:
The training data (train.csv) contains 785 columns, of which 1 column is a label column (label) for supervised learning tasks, and the remaining 784 columns are feature columns (pixel0 to pixel783).
The testing data (test.csv) contains 784 columns, including only feature columns (pixel0 to pixel783) and no label column.

## 4. Model Details
### Columns used as inputs in the final model:
All pixel columns (pixel0 to pixel783) from the dataset were used as input features.

### Column(s) used as target(s) in the final model:
The label column from the training dataset (train.csv) was used as the target variable, representing the actual digit (0–9) for supervised learning.

### Type of model:
Logistic Regression
Support Vector Machine (SVM)
Convolutional Neural Network (CNN)

### Software used to implement the model:
The models were implemented using Jupyter Notebook in Python.

### Version of the modeling software:
Python: 3

### Libraries used:
pandas (for data manipulation)
numpy (for numerical operations)
scikit-learn (for Logistic Regression and SVM)
tensorflow/Keras (for CNN)

### Hyperparameters or other settings of your model:
#### Logistic Regression:
Solver: lbfgs
Maximum iterations: 1000
Target: Flattened label array (np.argmax(y_train, axis=1))

#### Support Vector Machine (SVM):
Kernel: rbf
Regularization parameter (C): 10
Target: Flattened label array (np.argmax(y_train, axis=1))

#### Convolutional Neural Network (CNN):
Input shape: (28, 28, 1) (grayscale images reshaped to 28x28x1)
Layers:
Conv2D: 32 filters, kernel size (3, 3), activation function: ReLU
MaxPooling2D: Pool size (2, 2)
Flatten: Converts 2D feature maps into 1D feature vectors
Dense: 128 units, activation function: ReLU
Dense (Output): 10 units (for digits 0–9), activation function: Softmax
Loss function: categorical_crossentropy
Optimizer: Adam
Metrics: accuracy
Number of epochs: 10 (default, can be tuned)
Batch size: 32

## 5.Quantitative Analysis
### Metrics used to evaluate your final model
The models were evaluated using Accuracy and AUC (Area Under the Curve) as the performance metrics.
Accuracy:
Logistic Regression: 0.9188
SVM: 0.9799
CNN: 0.9898
AUC:
Logistic Regression: 0.9929
SVM: 0.9996
CNN: 0.9999

### Final values of metrics (training, validation, and test data)
| **Model**              | **Training Accuracy** | **Validation Accuracy** | **AUC**   |
|-------------------------|-----------------------|-------------------------|-----------|
| Logistic Regression     | 0.9250               | 0.9188                  | 0.9929    |
| SVM                     | 0.9832               | 0.9799                  | 0.9996    |
| CNN                     | 0.9965               | 0.9898                  | 0.9999    |

### Plots related to the data and final model
The following are charts related to model training and performance evaluation:
#### CNN model training and validation accuracy chart
Description: Shows the training and validation accuracy of the CNN model in each epoch.
Label: X-axis - Epochs, Y-axis - Accuracy.

#### CNN model training and validation loss function chart
Description: Shows the change of loss function between training and validation to ensure model convergence.
Label: X-axis - Epochs, Y-axis - Loss.

#### Confusion Matrix for Three Models
Logistic Regression: Shows the classification accuracy of the model on the validation set.
SVM: Shows the performance of SVM on the validation set.
CNN: Shows the classification results of the CNN model on the validation set.
Label: X-axis - Predicted Labels, Y-axis - True Labels.

## 6. Ethical Considerations
### Potential negative impacts of using the model
#### Math or software problems:
The model's performance heavily relies on the quality of the MNIST dataset and its pixel features. Inaccuracies may occur if the input data is noisy, incomplete, or poorly preprocessed.
The CNN model's complexity may lead to overfitting, especially when training on limited or biased data.

#### Real-world risks:
What/Who: Misclassification could occur for handwritten digits that are not similar to MNIST (e.g., real-world handwriting styles).
When/How: If deployed in scenarios requiring high precision, such as financial document processing or identity verification, errors may cause financial or operational issues.

### Potential uncertainties relating to the impacts of the model
#### Math or software problems:
The CNN model may produce unexpected results due to the randomness of weight initialization and hyperparameter selection.
Logistic Regression and SVM models are simpler but may underperform on more complex handwritten datasets.

#### Real-world risks:
The current model was trained on the MNIST dataset, which is clean and standardized. Performance may drop significantly when applied to real-world, non-standard handwriting.
Risks include incorrect predictions in critical environments, such as automated systems requiring human verification.

### Unexpected results
CNN performance: The CNN model achieved exceptionally high validation accuracy (0.9898) and AUC (0.9999), which may indicate overfitting due to limited data or insufficient validation techniques.
