# AIforTumorEvaluation
# AI Tutorial Practice

This repository contains code I practiced from a YouTube AI tutorial. The original tutorial can be found [here]((https://www.youtube.com/watch?v=z1PGJ9quPV8&t=583s)).

## Original Tutorial Information
- **Author**: https://www.youtube.com/@theadameubanks : Adam Eubanks
- **Link**: [YouTube Video](https://www.youtube.com/watch?v=z1PGJ9quPV8&t=583s)
- **Description**: This is an Ai tutorial based on real world tumor data. This AI is built on the model where it predicts whether a tumor is beniegn or Malignant. The code is short and easy to understand if you have a basic knowledge of AI-ML & Python. This helped me a lot to understand the basics of machine learning and Python as well. Shout out to Adam Eubanks.

## Modifications and Improvements
- I have added an extra line of code to it in the 6th cell. While running the original code the accuracy came upto 0.00000^n. So i properly called out the model and properly defined Tensorflow, keras. The updation is taken from the previous cell (5th cell) where tensorflow is imported. The updated code cell has been pasted below:
- model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

## How to Run
- Set up your Google Colab account and then download the cancer.csv file (https://github.com/AroraDrishti/IntroductionToDataScience) **Please find the link to download the .csv file in the link that i have pasted**
- Then connect to the colab and select the downloaded cancer.csv file and then run your code cell by cell.
- Please find the code below:


import pandas as pd
dataset = pd.read_csv('/cancer.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])

y = dataset["diagnosis(1=m, 0=b)"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


import tensorflow as tf
model = tf.keras.models.Sequential()


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=1000)


model.evaluate(x_test, y_test)



