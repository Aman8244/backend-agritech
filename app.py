from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
# python -m venv venv
# source venv/bin/activate
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import pandas as pd
data = pd.read_csv("animal_disease_dataset.csv")
# data["Animal"].unique()
replacement_dict = {
    'cow': 1,
    'buffalo':2,
    'sheep':3,
    'goat':4
}

# Use the .replace() method to perform the replacement
data['Animal'] = data['Animal'].replace(replacement_dict)

# data['Symptom 1'].unique()
replica_object = {
    'depression':10,
    'painless lumps':11,
    'loss of appetite':12,
    'difficulty walking':13,
    'lameness':14,
    'chills':15,
    'crackling sound':16,
    'sores on gums':17,
    'fatigue':18,
    'shortness of breath':19,
    'chest discomfort':20,
    'swelling in limb':21,
    'swelling in abdomen':22,
    'blisters on gums':23,
    'swelling in extremities':24,
    'swelling in muscle':25,
    'blisters on hooves':26,
    'blisters on tongue':27,
    'sores on tongue':28,
    'sweats':29,
    'sores on hooves':30,
    'blisters on mouth':31,
    'swelling in neck':32,
    'sores on mouth':33
}
data['Symptom 1']=data['Symptom 1'].replace(replica_object)
data['Symptom 2']=data['Symptom 2'].replace(replica_object)
data['Symptom 3']=data['Symptom 3'].replace(replica_object)
replica_obj_disease = {
    'pneumonia':50,
    'lumpy virus':51,
    'blackleg':52,
    'foot and mouth':53,
    'anthrax':54
}
data['Disease'] = data['Disease'].replace(replica_obj_disease)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
df2 = data

# Define your features (X) and target variable (y)
X = df2[['Animal', 'Age', 'Temperature','Symptom 1', 'Symptom 2', 'Symptom 3']]
y = df2['Disease']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Categorical Naive Bayes classifier
nb_classifier = CategoricalNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)


@app.route('/',methods=['GET'])
def Home():
    return "<h1>Hello</h1>"
@app.route('/predict', methods=['POST'])
def predict():
    # input_data = data[['Animal', 'Age', 'Temperature','Symptom 1', 'Symptom 2', 'Symptom 3']].head(2)
    # Perform preprocessing on input data if needed
    input_data = request.json
    processed_data = pd.DataFrame([[input_data["Animal"],input_data["Age"],input_data["Temperature"],input_data["Symptom 1"],input_data["Symptom 2"],input_data["Symptom 3"]]])

    prediction = nb_classifier.predict(processed_data)
    print(prediction[0])
    # print(processed_data)

    return "working"

if __name__ == '__main__':
    app.run(debug=True)
