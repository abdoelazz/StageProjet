from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import uvicorn
import nltk
import threading
import time
from nltk.corpus import stopwords
import string

# Initialize NLTK and download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Load models and vectorizer
models = {
    "Naive Bayes": pickle.load(open('Naive Bayes_model.pkl', 'rb')),
    "Logistic Regression": pickle.load(open('Logistic Regression_model.pkl', 'rb')),
    "SVM": pickle.load(open('SVM_model.pkl', 'rb')),
    "Random Forest": pickle.load(open('Random Forest_model.pkl', 'rb'))
}
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# File to store predictions for re-training
data_file = 'incidents_reseau.csv'

# Helper function to preprocess text
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict")
def predict(request: Request,
            description: str = Form(...),
            model_name: str = Form(...)):
    # Preprocess the input
    preprocessed_description = preprocess_text(description)
    
    # Vectorize the input
    data = vectorizer.transform([preprocessed_description])
    
    # Predict using the selected model
    model = models.get(model_name)
    prediction = model.predict(data)[0]
    
    # Save the input and prediction for future training
    new_data = pd.DataFrame([[description, prediction]], columns=["Description", "Type d'incident"])
    try:
        existing_data = pd.read_csv(data_file)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data
    updated_data.to_csv(data_file, index=False)
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "prediction_text": f"Le type d'incident prédit est {prediction}"
    })

def retrain_models():
    
    while True:
        time.sleep(60)  
        try:
            
            data = pd.read_csv(data_file)
            data['Description'] = data['Description'].apply(preprocess_text)
            X = vectorizer.transform(data['Description'])
            y = data["Type d'incident"]
            for name, model in models.items():
                model.fit(X, y)
                pickle.dump(model, open(f"{name}_model.pkl", "wb"))
            
            print("Models successfully re-trained with new data.")
        except FileNotFoundError:
            print("No data available for re-training. Skipping this cycle.")
        except Exception as e:
            print(f"Error during retraining: {e}")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=retrain_models, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
