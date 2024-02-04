# Diabetes-Prediction-ML
Description:
->This repository contains a machine learning model built with TensorFlow to predict the likelihood of diabetes based on certain medical indicators. The model is trained on the Pima Indians Diabetes Database.

Dependencies:
->Python 3.x
->TensorFlow
->NumPy
->Pandas
->Matplotlib
->Scikit-learn
->imbalanced-learn (for RandomOverSampler)

Model Architecture:
The model architecture consists of a neural network with:
->Input layer with the same number of neurons as input features
->Two hidden layers with 16 neurons each, activated by ReLU (Rectified Linear Unit) activation function
->Output layer with one neuron activated by sigmoid function for binary classification

Usage:
1.Clone the repository:
-> git clone https://github.com/your-username/Diabetes-Prediction-ML.git
2.Install dependencies:
-> pip install -r diabetes_requirements.txt
3.Run the model:
-> python diabetes_prediction.py

File Structure:
->diabetes.csv: Dataset containing medical indicators and diabetes outcomes.
->diabetes_prediction.py: Python script containing code to preprocess data, train the model, and evaluate its performance.
->README.md: Markdown file providing an overview of the repository, instructions for usage, and details about the model and dependencies.

How to Run:
->Ensure you have Python and the required libraries installed.
->Clone the repository to your local machine.
->Navigate to the repository directory.
->Install dependencies using pip install -r requirements.txt.
->Run the diabetes_prediction.py script using python diabetes_prediction.py.

Note:
->The dataset used in this model is the Pima Indians Diabetes Database, available in diabetes.csv.
->The model is trained using TensorFlow's Sequential API and evaluated using binary cross-entropy loss and accuracy metrics.
