# LLM MODEL FOR TEXT DETECTION (Improved Version)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load datasets
train = pd.read_csv("F:\\Downloads\\train_v2_drcat_02.csv")
train_1 = pd.read_csv("F:\\Downloads\\Training_Essay_Data.csv")

# Rename for consistency
train_1.rename(columns={'generated': 'label'}, inplace=True)

# Drop NaN values
train.dropna(inplace=True)
train_1.dropna(inplace=True)

# Drop unnecessary columns
train = train.drop(columns=['prompt_name', 'source', 'RDizzl3_seven'])

# Combine datasets
combined_data = pd.concat([train, train_1], ignore_index=True)

# TF-IDF Vectorizer (memory-safe)
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(combined_data['text']).astype(np.float32)
y = combined_data['label'].values.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=144, stratify=y
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# Convert sparse matrices to dense only when batching
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train.toarray(), y_train)
).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test.toarray(), y_test)
).batch(32)

# Build neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# Train with class weights
history = model.fit(train_dataset,
                    epochs=15,
                    validation_data=test_dataset,
                    class_weight=class_weights_dict,
                    callbacks=[early_stop, checkpoint],
                    verbose=1)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Predictions
y_pred_prob = model.predict(X_test.toarray())
y_pred = (y_pred_prob > 0.7).astype(int)  # stricter threshold

# Reports
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save tokenizer
with open('tfidf_tokenizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save final model
model.save('text_classification_model.h5')