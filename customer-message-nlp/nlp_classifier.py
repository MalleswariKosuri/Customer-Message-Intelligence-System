#Step 0 : Installing and Importing required packages

import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Step 1 — Load and inspect the language data

data = pd.read_csv("support_messages.csv")

print(data.head())
print("Number of rows in the data: ",len(data))
print(data["category"].value_counts())


#Step 2 — Split training and test data

X = data["message"]
y= data["category"]

X_train, X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.25,
    random_state = 42,
    stratify=y 
)
print("Training_messages : \n", X_train)
print("Traning category: \n", y_train)

#Step 3: Build Version A: unigrams only

unigram_model = Pipeline([
    ("Tfid Vectorizer",
     TfidfVectorizer
     (
        lowercase = True,
        ngram_range = (1,1),

    )),
    ("Classifier",
     LogisticRegression(max_iter = 1000))
])

unigram_model.fit(X_train,y_train)

#Step 4  Evaluate the model using unigram

unigram_predictions = unigram_model.predict(X_test)
unigram_accuracy = accuracy_score(y_test,unigram_predictions)

print("\n Unigram accuracy: ",round(unigram_accuracy,3))

results = pd.DataFrame({
    "message": X_test,
    "Actual": y_test,
    "unigram_predicted": unigram_predictions,
})

print("\n Test Predictions :")

print(results.to_string(index = False))


# Step 5 — Build Version B: unigrams + bigrams

bigram_model = Pipeline([(
    "Tfidf Vectorizer",
    TfidfVectorizer(
        lowercase = True,
        ngram_range = (1,2)
    )
    ),

    ("Classifier",
   LogisticRegression(max_iter = 1000)
   )
   ])

bigram_model.fit(X_train,y_train)

#Step 5  Evaluate the model using unigram + Bigram

bigram_prediction = bigram_model.predict(X_test)

bigram_accuracy = accuracy_score(y_test,bigram_prediction)

result = pd.DataFrame({
    "message" : X_test,
    "Actuals" : y_test,
    "Predicted" : bigram_prediction
    })

print("\n Test prediction Bigram Model")
print(round(bigram_accuracy,3))
print(result.to_string(index = False))

#Step 6 Graphical representation of unigram and bigram model accuracy

models = ["Unigram model", "uni + Bigram models"]

accuracy = [unigram_accuracy, bigram_accuracy]

bars = plt.bar(
    models,
    accuracy,
    color=["pink", "green"],
    width = 0.5
)

plt.ylabel("Accuracy")
plt.title("Accuracy of unigram and uni-bigram models")
plt.show()


# Step 7 : Test both models on the same new messages

new_messges = [
    "Money disappeared from my card after checkout",
    "I cannot get into my account",
    "My parcel should have arrived yesterday",
    "The app closes as soon as I open it",
    "I need my money back",
    "The sign in code is missing"
]

unigram_new_predict = unigram_model.predict(new_messges)
bigram_new_predict = bigram_model.predict(new_messges)

for message,uni,bi in zip(new_messges,unigram_new_predict,bigram_new_predict):
    print("\n message", message)
    print("\n unigram prediction: ", uni)
    print("\n bigram prediction: ", bi)


# Step 8 : Checking how confident model is about in each possible category

probabilities = bigram_model.predict_proba(new_messges)
classes = bigram_model.classes_

for message,scores in zip(new_messges,probabilities):
    print("\n message: ", message)

    ranked = sorted(
        zip(classes, scores),
        key = lambda item : item[1],
        reverse = True      
    )
    for category, score in ranked:
        print(category, "=" , round(score,3))

#Step 9 : The NLP Stress Test

tricky_message = [
    "I can login but my payment keeps failing",
    "My order arrived but the device does not work",
    "Great, another package that never arrived",
    "payment deduct ayyindi but order confirm avvaledu",
    "I need help with something"]

tricky_predictions = bigram_model.predict(tricky_message)

for message,tricky in zip(tricky_message,tricky_predictions):
    print(message , " - ", tricky)