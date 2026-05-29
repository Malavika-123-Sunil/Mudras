import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle

# Load dataset
data = pd.read_csv("dataset.csv", header=None)

# Features (coordinates)
# X gets all clues (all rows, all columns EXCEPT the last one)
# Before , its Row ( : ) : starting from default 0 and ending also default all row and 
# After , its Coln (:-1): start: default 0 and stop: -1(excluding last coln)
X = data.iloc[:, :-1]   

# Labels (mudra names)
y = data.iloc[:, -1]    # y gets the target answer (all rows, ONLY the last coln) 

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
with open("mudra_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")