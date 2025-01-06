import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# 1. Załaduj dane
def load_data():
    # Załaduj pliki z danymi (upewnij się, że ścieżki do plików są poprawne)
    phishing_data = pd.read_csv("phishing_urls.csv")  # Plik z adresami phishingowymi
    safe_data = pd.read_csv("safe_urls.csv")  # Plik z bezpiecznymi adresami
    phishing_data['label'] = 1
    safe_data['label'] = 0

    # Zbalansowanie danych przez oversampling
    phishing_data = resample(phishing_data, replace=True, n_samples=len(safe_data), random_state=42)

    # Połącz dane w jeden zbiór
    data = pd.concat([phishing_data, safe_data]).drop_duplicates(subset='url').reset_index(drop=True)
    return data['url'], data['label']

urls, labels = load_data()

# 2. Tokenizacja i kodowanie adresów URL
max_words = 20000  # Zwiększona liczba unikalnych tokenów
max_len = 300      # Zwiększona maksymalna długość sekwencji

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(urls)

# Debugowanie: sprawdzenie przykładowych tokenów
print("Example tokens:", list(tokenizer.word_index.items())[:10])

# Zamień adresy URL na sekwencje liczb
sequences = tokenizer.texts_to_sequences(urls)
x_data = pad_sequences(sequences, maxlen=max_len)

# Kodowanie etykiet
encoder = LabelEncoder()
y_data = encoder.fit_transform(labels)

# Podział danych na zbiory treningowy i testowy (50/50 dla lepszego testowania modelu)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42)

# 3. Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Początkowa definicja wymiaru słownika
vocab_size = len(tokenizer.word_index) + 1  # Obejmuje 0, który jest w tokenizacji
embedding_dim = 100  # Zdefiniuj wymiar embeddingu, np. 100

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# Trenowanie modelu z EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test),
                    class_weight=class_weight_dict, callbacks=[early_stopping], verbose=1)


# 7. Ocena modelu
evaluation_results = model.evaluate(x_test, y_test)
print(f"Evaluation Results: {evaluation_results}")

# 8. Analiza wyników

# Przewidywania na zbiorze testowym
y_pred = (model.predict(x_test) > 0.5).astype("int32")

# Wyświetlanie metryk
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Wizualizacja wyników
# Accuracy per epoch
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss per epoch
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ROC Curve
y_pred_prob = model.predict(x_test)  # Uzyskujemy prawdopodobieństwa, a nie klasy
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 9. Predykcja nowych adresów URL
def predict_url(url):
    sequence = tokenizer.texts_to_sequences([url])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return "Phishing" if prediction[0] > 0.5 else "Safe"

# Przykład użycia
test_urls = [
    "http://example.com/phishing_link",
    "http://google.com"
]

for test_url in test_urls:
    print(f"URL: {test_url} - {predict_url(test_url)}")
