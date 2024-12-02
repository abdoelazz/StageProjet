import re
import unicodedata
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import uvicorn
import nltk
from nltk.corpus import stopwords
import string
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer("french")
# Initialize NLTK and download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

app = FastAPI()
templates = Jinja2Templates(directory='templates')



# Load models dynamically from the models directory
models = {
    "Naive Bayes": pickle.load(open(os.path.join("models", "Naive Bayes_model.pkl"), 'rb')),
    "Logistic Regression": pickle.load(open(os.path.join("models", "Logistic Regression_model.pkl"), 'rb')),
    "SVM": pickle.load(open(os.path.join("models", "SVM_model.pkl"), 'rb')),
    "Random Forest": pickle.load(open(os.path.join("models", "Random Forest_model.pkl"), 'rb'))
}

# Load the vectorizer from the vectorizer directory
vectorizer_path = os.path.join("vectorizer", "vectorizer.pkl")
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# File to store predictions for re-training
data_file = 'incidents_reseau.csv'

# Helper function to preprocess text
def preprocess_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Supprimer les stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # supprimer les nombres
    text = re.sub(r'\d+', '', text)
    # normaliser les lettres qui ont des accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Appliquer le stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    
    return text

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

input_folder = "./input"
output_folder = "./predictions"

@app.post("/predict")

def process_files(request: Request, model_name: str = Form(...)):
    # Step 1: Check for files in the input folder
    try:
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv') and not f.endswith('predicted.csv')]
        if not csv_files:
            return templates.TemplateResponse("home.html", {
                "request": request,
                "error_text": "No CSV files found in the input folder."
            })
    except Exception as e:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error_text": f"Error accessing the input folder: {e}"
        })
    # Step 2: Process each file
    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        output_path = os.path.join(output_folder, f"predicted_{csv_file}")
        output_df = pd.DataFrame()
        # Read the CSV file
        input_df = pd.read_csv(input_path)
        if "Description" not in input_df.columns:
            continue  # Skip files without a "Description" column
        descriptions = input_df["Description"].tolist()
        # Preprocess and predict
        preprocessed_descriptions = [preprocess_text(desc) for desc in descriptions]
        data = vectorizer.transform(preprocessed_descriptions)
        model = models.get(model_name)
        predictions = model.predict(data)
        output_df["Description"] = input_df["Description"]
        # Save predictions in a new column
        output_df["Type d'incident"] = predictions
        
        # Save the updated file to the output folder
        output_df.to_csv(output_path, index=False)
        new_file_name = csv_file.replace('.csv', '_predicted.csv')
        trained_file_path = os.path.join(input_folder, new_file_name)
        os.rename(input_path, trained_file_path)
    
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "prediction_text": f"All files processed. Results saved to '{output_folder}'.",
        "model_name" : model_name
    })
@app.post("/retrain")
def retrain_models(request: Request, model_name: str = Form(...)):
    try:
        csv_files = [
            f for f in os.listdir(output_folder)
            if f.endswith('.csv') and not f.endswith('trained.csv') and not f.endswith('validated.csv') 
        ]
        if not csv_files:
            return templates.TemplateResponse("home.html", {
                "request": request,
                "error_text": "No CSV files found in the training folder."
            })
    except Exception as e:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error_text": f"Error accessing the output folder: {e}"
        })
    
    # Load existing data
    existing_data = pd.read_csv(data_file)
    updated_data = existing_data
    print("retraining_data : \n")
    # Merge new data
    for csv_file in csv_files:
        print(csv_file)
        newdata_path = os.path.join(output_folder, csv_file)
        newdata_df = pd.read_csv(newdata_path)
        
        # Check for required columns
        if "Description" not in newdata_df.columns or "Type d'incident" not in newdata_df.columns:
            continue  # Skip files without required columns
        
        # Concatenate new data to the existing data
        updated_data = pd.concat([updated_data, newdata_df], ignore_index=True)
        # Rename the file to mark it as processed
        new_file_name = csv_file.replace('.csv', '_trained.csv')
        trained_file_path = os.path.join(output_folder, new_file_name)
        os.rename(newdata_path, trained_file_path)

    # Preprocess the data
    updated_data['Description'] = updated_data['Description'].apply(preprocess_text)
    X = vectorizer.transform(updated_data['Description'])
    y = updated_data["Type d'incident"]
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Get the current model
    old_model = models.get(model_name)
    if old_model is None:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error_text": f"Model '{model_name}' not found."
        })
    
    # Evaluate the old model
    y_val_pred_old = old_model.predict(X_val)
    old_model_report = classification_report(y_val, y_val_pred_old, output_dict=True)
    old_model_accuracy = old_model_report["accuracy"]
    
    # Train the new model on the training set
    new_model = old_model.__class__()  # Create a new instance of the same model
    new_model.fit(X_train, y_train)
    
    # Evaluate the new model
    y_val_pred_new = new_model.predict(X_val)
    new_model_report = classification_report(y_val, y_val_pred_new, output_dict=True)
    new_model_accuracy = new_model_report["accuracy"]
    pickle.dump(new_model, open(os.path.join("models", f"new_{model_name}_model.pkl"), "wb"))
    return templates.TemplateResponse("retraining_report.html", {
        "request": request,
        "model_name": model_name,
        "old_model_accuracy": old_model_accuracy,
        "old_model_report": old_model_report,
        "new_model_accuracy": new_model_accuracy,
        "new_model_report": new_model_report
    })

@app.post("/savemodel")
def save_model(request: Request, model_name: str = Form(...)):
    try:
        csv_files = [
            f for f in os.listdir(output_folder)
            if f.endswith('trained.csv')
        ]
        if not csv_files:
            return templates.TemplateResponse("home.html", {
                "request": request,
                "error_text": "No CSV files found in the output folder to save."
            })
    except Exception as e:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error_text": f"Error accessing the output folder: {e}"
        })
    existing_data = pd.read_csv(data_file)
    all_new_data = existing_data
    print("saving training_data : \n")
    for csv_file in csv_files:
        print(csv_file)
        newdata_path = os.path.join(output_folder, csv_file)
        newdata_df = pd.read_csv(newdata_path)
        # Concatenate new data to the existing data
        all_new_data = pd.concat([all_new_data, newdata_df], ignore_index=True)
        new_file_name = csv_file.replace('.csv', '_validated.csv')
        trained_file_path = os.path.join(output_folder, new_file_name)
        os.rename(newdata_path, trained_file_path)
    all_new_data.to_csv(data_file, index=False)
    # Paths for the new and old models
    new_model_path = os.path.join("models", f"new_{model_name}_model.pkl")
    old_model_path = os.path.join("models", f"{model_name}_model.pkl")
    
    # Generate a timestamp for the backup file
    timestamp = datetime.now().strftime("%Y%m%d")
    old_backup_path = os.path.join("models", f"old_{model_name}_{timestamp}_model.pkl")

    try:
        # Check if the new model exists
        if not os.path.exists(new_model_path):
            return templates.TemplateResponse("saved.html", {
                "request": request,
                "prediction_text": f"No new model found for '{model_name}'. Retrain the model first."
            })

        # Rename the old model to a timestamped backup version
        if os.path.exists(old_model_path):
            os.rename(old_model_path, old_backup_path)
        
        # Rename the new model to replace the original
        os.rename(new_model_path, old_model_path)
        
        return templates.TemplateResponse("saved.html", {
            "request": request,
            "message": f"New model for '{model_name}' has been saved successfully. "
                               f"The old model has been backed up as '{old_backup_path}'."
        })
    except Exception as e:
        return templates.TemplateResponse("saved.html", {
            "request": request,
            "message": f"Error saving the model: {e}"
        })


@app.post("/dontsavemodel")
def dontsave_model(request: Request, model_name: str = Form(...)):
    try:
        csv_files = [
            f for f in os.listdir(output_folder)
            if f.endswith('trained.csv')
        ]
        if not csv_files:
            return templates.TemplateResponse("home.html", {
                "request": request,
                "error_text": "No CSV files found in the training folder to delete."
            })
    except Exception as e:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error_text": f"Error accessing the input folder: {e}"
        })
    print("not saving training_data : \n")
    for csv_file in csv_files:
        print(csv_file)
        newdata_path = os.path.join(output_folder, csv_file)
        new_file_name = csv_file.replace('.csv', '_not_validated.csv')
        trained_file_path = os.path.join(output_folder, new_file_name)
        os.rename(newdata_path, trained_file_path)

    # Path for the new model
    new_model_path = os.path.join("models", f"new_{model_name}_model.pkl")

    try:
        # Check if the new model exists
        if not os.path.exists(new_model_path):
            return templates.TemplateResponse("saved.html", {
                "request": request,
                "prediction_text": f"No new model found for '{model_name}'. Retrain the model first."
            })
        
        # Delete the new model file
        os.remove(new_model_path)
        
        return templates.TemplateResponse("saved.html", {
            "request": request,
            "prediction_text": f"The new model for '{model_name}' has been successfully deleted."
        })
    except Exception as e:
        return templates.TemplateResponse("saved.html", {
            "request": request,
            "prediction_text": f"Error deleting the new model: {e}"
        })

    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
