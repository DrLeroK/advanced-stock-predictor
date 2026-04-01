"""
Return magnitude regressor with Huber loss
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Bidirectional, LSTM, Dense, Dropout,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class HuberRegressor:
    """
    Neural network for predicting return magnitude with Huber loss
    """
    
    def __init__(self, sequence_length=60, n_features=50, 
                 lstm_units=128, dropout_rate=0.3, learning_rate=0.001,
                 huber_delta=1.0):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.huber_delta = huber_delta
        self.model = None
    
    def build(self):
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Convolutional layers
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', 
                       activation='relu', kernel_regularizer=l2(0.001))(inputs)
        conv1 = LayerNormalization()(conv1)
        conv1 = Dropout(self.dropout_rate)(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=3, padding='same',
                       activation='relu', kernel_regularizer=l2(0.001))(conv1)
        conv2 = LayerNormalization()(conv2)
        conv2 = Dropout(self.dropout_rate)(conv2)
        
        # Bidirectional LSTM
        bilstm = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                                     kernel_regularizer=l2(0.001)))(conv2)
        bilstm = LayerNormalization()(bilstm)
        bilstm = Dropout(self.dropout_rate)(bilstm)
        
        # Attention
        attention = MultiHeadAttention(num_heads=4, key_dim=self.lstm_units // 2)(bilstm, bilstm)
        attention = LayerNormalization()(attention)
        
        # Global pooling
        pooled = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(pooled)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense1 = LayerNormalization()(dense1)
        
        dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense1)
        dense2 = Dropout(self.dropout_rate)(dense2)
        
        # Output layer
        output = Dense(1, activation='linear', name='return')(dense2)
        
        self.model = Model(inputs=inputs, outputs=output)
        
        # Use Huber loss for outlier robustness
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(delta=self.huber_delta),
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()