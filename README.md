# Next Word Prediction with LSTM

## Overview
This project implements a **Next Word Prediction Model** using **LSTM (Long Short-Term Memory) networks**. The model is trained on the Shakespearean text dataset to predict the next word in a given sequence. It utilizes **NLTK for data processing, TensorFlow/Keras for training, and Streamlit for a user-friendly web interface**.

---

## Features
- **Data Preprocessing:** Tokenization, sequence padding, and text normalization.
- **LSTM Model Training:** Implemented with **Early Stopping** to prevent overfitting.
- **Prediction Function:** Generates the most probable next word for an input sequence.
- **Model Saving & Loading:** Trained model and tokenizer are saved for future use.
- **Streamlit Web App:** Interactive interface to enter text and predict the next word.

---

## Project Structure
```
NextWord-LSTM/
│-- dataset/                   # Contains processed text files
│   ├── hamlet.txt             # Shakespeare's Hamlet text file
│-- models/                    # Stores trained models and tokenizer
│   ├── next_word_lstm.h5      # Saved LSTM model
│   ├── tokenizer.pickle       # Saved tokenizer
│-- app/                       # Streamlit application
│   ├── app.py                 # Streamlit UI for word prediction
│-- README.md                  # Documentation (this file)
│-- requirements.txt           # Required Python packages
│-- .gitignore                 # Git ignore file
```

---

## Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yasirwali1052/NextWord-LSTM.git
cd NextWord-LSTM
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.8+** installed. Then, install the required packages:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```sh
streamlit run app/app.py
```

---

## Dataset Collection & Preprocessing
- **Dataset:** Uses Shakespeare's Hamlet text from the **NLTK Gutenberg Corpus**.
- **Tokenization:** Converts words into numerical indexes.
- **Sequence Padding:** Ensures consistent input length for the LSTM model.

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-hamlet.txt')
with open('dataset/hamlet.txt', 'w') as file:
    file.write(data)
```

---

## Model Architecture
The model consists of:
1. **Embedding Layer:** Converts words into dense vector representations.
2. **LSTM Layers:** Two stacked LSTM layers with dropout for regularization.
3. **Dense Layer:** Applies softmax activation for multi-class classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len-1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
```

**Compilation & Training**:
```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])
```

---

## Model Saving & Loading
After training, the model and tokenizer are saved for later use.
```python
import pickle
model.save('models/next_word_lstm.h5')
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

To load and use the model:
```python
from tensorflow.keras.models import load_model
model = load_model('models/next_word_lstm.h5')
```

---

## Next Word Prediction Function
```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
```

---

## Streamlit Web Application
An interactive UI for testing the model in real-time.

### **Usage**:
1. Enter a sequence of words.
2. Click **Predict Next Word**.
3. The predicted next word is displayed.

```python
import streamlit as st
st.title("Next Word Prediction With LSTM")
input_text = st.text_input("Enter a sequence of words", "To be or not to")
if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
```



## Deployment
The model can be deployed on **Render, Hugging Face Spaces, or AWS Lambda**.

For **Render Deployment**:
1. Create a `requirements.txt` file with all dependencies.
2. Deploy using a **Flask API** or Streamlit.

### 🚀 If you found this useful, don't forget to 🌟 star the repository! 🚀

