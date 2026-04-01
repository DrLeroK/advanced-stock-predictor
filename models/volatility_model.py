"""
Volatility prediction model
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class VolatilityPredictor:
    """
    Neural network for predicting volatility regimes
    """
    
    def __init__(self, sequence_length=60, n_features=50, 
                 lstm_units=64, dropout_rate=0.3, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
    
    def build(self):
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM for volatility patterns
        lstm1 = LSTM(self.lstm_units, return_sequences=True,
                     kernel_regularizer=l2(0.001))(inputs)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        lstm2 = LSTM(self.lstm_units // 2, return_sequences=False,
                     kernel_regularizer=l2(0.001))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        
        # Dense layers
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(lstm2)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        # Output for volatility (regression)
        output = Dense(1, activation='linear', name='volatility')(dense1)
        
        self.model = Model(inputs=inputs, outputs=output)
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()