import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from zipfile import ZipFile
import urllib.request
from io import BytesIO
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Download the ZIP file and extract the CSV file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

with urllib.request.urlopen(url) as response:
    with ZipFile(BytesIO(response.read())) as zip_file:
        # Assuming the CSV file is named 'SMSSpamCollection'
        with zip_file.open('SMSSpamCollection') as csv_file:
            sms_data = pd.read_csv(csv_file, sep='\t', names=['label', 'message'])

# Text preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

sms_data['message'] = sms_data['message'].apply(clean_text)

# Plotting pie chart for spam and non-spam messages
spam_counts = sms_data['label'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(spam_counts, labels=['Non-Spam', 'Spam'], autopct='%1.1f%%', colors=['blue', 'red'])
plt.title('Distribution of Spam and Non-Spam Messages')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sms_data['message'], sms_data['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the classifier
classifier = MultinomialNB()

# Define the grid of parameters to search
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

# Grid Search for hyperparameter tuning
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_tfidf, y_train)

# Train the classifier with the best parameters
best_classifier = grid_search.best_estimator_
best_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
predictions = best_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Print results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
