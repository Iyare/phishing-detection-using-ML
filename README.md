# A Content-Based Phishing Detection Project using ML

## Project Summary

This project adopted a supervised learning method in training several Scikit-learn models commonly used in classification projects. These models include: Gaussian Naive Bayes, Support Vector Machines, Decision Trees, Random Forest, AdaBoost, Neural Network, and K-Neighbours. After training the models with labeled data, models were tested with a seperate test data. Results detailing the accuracy, recall and precision values were obtained for each model. The best performing model was the Random Forest with a precision value of 81%, 69% recall, and 98% accuracy. The web interface of this application allows users to use the trained models to detect phishing or legitimate websites based on their HTML content NOT URL features like URL length, etc.

## Project Requirements

This project is written primarily in python and thus utilizes several libraries/modules as stated below:

1. Scikit-learn
2. BeautifulSoup4
3. Matplotlib
4. Numpy
5. Pandas
6. Requests
7. Streamlit
8. Urllib3

## Process

The project uses supervised learning approach to train several ML models to detect phishing.

