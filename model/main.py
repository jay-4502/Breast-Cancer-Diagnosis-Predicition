import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import logging

def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # Ensure X has column names
  X.columns = [f'feature_{i}' for i in range(X.shape[1])]
  
  # Ensure data types of features match the training data
  X = X.astype(float)

  # scale the data
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

    # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
    )
  
  # train the model
  model = LogisticRegression()
  model.fit(X_train, y_train)
  
  # test model
  y_pred = model.predict(X_test)

  cfm=confusion_matrix(y_test, y_pred)
  labels = ['Benign', 'Malignant']
  sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()

  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

def save_model_and_scaler(model, scaler):
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def main():
  try:
        logging.basicConfig(level=logging.INFO)
        
        data = get_clean_data()
        model, scaler = create_model(data)

        save_model_and_scaler(model, scaler)

        logging.info("Model and scaler saved successfully.")

  except Exception as e:
      logging.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
  main()