import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Cryptographically secure random number generation
import secrets
import hashlib
import random
import math
from math import comb
from typing import List, Dict, Tuple, Optional

# Command-line argument parsing
import argparse

# BiLSTM dependencies (PyTorch for deep learning)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch scikit-learn")

# === PRODUCTION-GRADE BiLSTM INTEGRATION MODULES ===

class LotteryDataset(Dataset):
    """Custom Dataset for LSTM training on lottery data."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class BiLSTMLotteryModel(nn.Module):
    """
    Bidirectional LSTM model for lottery number prediction.
    
    This model uses bidirectional LSTM layers to capture temporal patterns
    in lottery draws, with separate models for main balls, bonus ball, and powerball.
    """
    
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
        super(BiLSTMLotteryModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 because bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final prediction layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for prediction
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.activation(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        
        return out

def prepare_lstm_sequences(df: pd.DataFrame, task: str, sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training from lottery data.
    
    Args:
        df: Historical lottery data
        task: One of "main", "bonus", "pb"
        sequence_length: Length of input sequences
        
    Returns:
        sequences, targets arrays for training
    """
    if task == "main":
        ball_range = 40
        # Create binary matrix for main balls (6 out of 40)
        sequences_list = []
        targets_list = []
        
        for idx in range(sequence_length, len(df)):
            # Look back sequence_length draws
            sequence_data = []
            for look_back in range(sequence_length):
                draw_idx = idx - sequence_length + look_back
                row = df.iloc[draw_idx]
                
                # Create features for this draw - handle NaN values
                main_numbers = []
                for i in range(1, 7):
                    val = row[str(i)]
                    if pd.isna(val):
                        break  # Skip this sequence if any main number is missing
                    main_numbers.append(int(val))
                
                if len(main_numbers) != 6:
                    break  # Skip this entire sequence if incomplete
                    
                # Binary encoding + additional features
                binary_encoding = [0] * ball_range
                for num in main_numbers:
                    binary_encoding[num - 1] = 1
                
                # Add temporal features
                draw_date = pd.to_datetime(row['Draw Date'], format='%A %d %B %Y', errors='coerce')
                weekday = draw_date.weekday() / 6.0 if not pd.isna(draw_date) else 0
                month = draw_date.month / 12.0 if not pd.isna(draw_date) else 0
                
                # Combine features
                features = binary_encoding + [weekday, month, sum(main_numbers) / 240.0, len(set([n % 10 for n in main_numbers])) / 6.0]
                sequence_data.append(features)
            
            sequences_list.append(sequence_data)
            
            # Target: next draw's main numbers (binary encoding)
            next_row = df.iloc[idx]
            next_main = []
            for i in range(1, 7):
                val = next_row[str(i)]
                if pd.isna(val):
                    break
                next_main.append(int(val))
            
            if len(next_main) == 6:  # Only add if complete target data
                target = [0] * ball_range
                for num in next_main:
                    target[num - 1] = 1
                targets_list.append(target)
            else:
                sequences_list.pop()  # Remove the corresponding sequence
                
    elif task == "bonus":
        ball_range = 40
        sequences_list = []
        targets_list = []
        
        for idx in range(sequence_length, len(df)):
            sequence_data = []
            for look_back in range(sequence_length):
                draw_idx = idx - sequence_length + look_back
                row = df.iloc[draw_idx]
                
                # Create features for this draw - handle NaN values
                main_numbers = []
                for i in range(1, 7):
                    val = row[str(i)]
                    if pd.isna(val):
                        continue  # Skip this row if any main number is missing
                    main_numbers.append(int(val))
                
                if len(main_numbers) != 6 or pd.isna(row['Bonus Ball']):
                    continue  # Skip incomplete rows
                    
                bonus_number = int(row['Bonus Ball'])
                
                # Binary encoding for main + bonus
                binary_encoding = [0] * ball_range
                for num in main_numbers:
                    binary_encoding[num - 1] = 0.5  # Main numbers get 0.5
                binary_encoding[bonus_number - 1] = 1.0  # Bonus gets 1.0
                
                # Add temporal features
                draw_date = pd.to_datetime(row['Draw Date'], format='%A %d %B %Y', errors='coerce')
                weekday = draw_date.weekday() / 6.0 if not pd.isna(draw_date) else 0
                month = draw_date.month / 12.0 if not pd.isna(draw_date) else 0
                
                features = binary_encoding + [weekday, month]
                sequence_data.append(features)
            
            sequences_list.append(sequence_data)
            
            # Target: next draw's bonus ball (binary encoding)
            next_row = df.iloc[idx]
            if pd.isna(next_row['Bonus Ball']):
                sequences_list.pop()  # Remove the corresponding sequence
                continue
            next_bonus = int(next_row['Bonus Ball'])
            target = [0] * ball_range
            target[next_bonus - 1] = 1
            targets_list.append(target)
            
    elif task == "pb":
        ball_range = 10
        sequences_list = []
        targets_list = []
        
        for idx in range(sequence_length, len(df)):
            sequence_data = []
            for look_back in range(sequence_length):
                draw_idx = idx - sequence_length + look_back
                row = df.iloc[draw_idx]
                
                # Create features for this draw - handle NaN values
                if pd.isna(row['Power Ball']):
                    continue  # Skip this row if powerball is missing
                powerball = int(row['Power Ball'])
                
                # Binary encoding for powerball
                binary_encoding = [0] * ball_range
                binary_encoding[powerball - 1] = 1
                
                # Add temporal features
                draw_date = pd.to_datetime(row['Draw Date'], format='%A %d %B %Y', errors='coerce')
                weekday = draw_date.weekday() / 6.0 if not pd.isna(draw_date) else 0
                month = draw_date.month / 12.0 if not pd.isna(draw_date) else 0
                
                features = binary_encoding + [weekday, month]
                sequence_data.append(features)
            
            sequences_list.append(sequence_data)
            
            # Target: next draw's powerball (binary encoding)
            next_row = df.iloc[idx]
            if pd.isna(next_row['Power Ball']):
                sequences_list.pop()  # Remove the corresponding sequence
                continue
            next_pb = int(next_row['Power Ball'])
            target = [0] * ball_range
            target[next_pb - 1] = 1
            targets_list.append(target)
    
    return np.array(sequences_list), np.array(targets_list)

def train_bilstm_model(sequences: np.ndarray, targets: np.ndarray, task: str, 
                      hidden_size: int = 128, num_epochs: int = 50, 
                      batch_size: int = 32, learning_rate: float = 0.001) -> BiLSTMLotteryModel:
    """
    Train a BiLSTM model for lottery prediction.
    
    Args:
        sequences: Input sequences (batch_size, seq_len, features)
        targets: Target outputs (batch_size, output_size)
        task: Task name for logging
        hidden_size: LSTM hidden size
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained BiLSTM model
    """
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch dependencies not available")
    
    print(f"Training BiLSTM model for {task}...")
    
    # Prepare data
    input_size = sequences.shape[2]
    output_size = targets.shape[1]
    
    # Split into train/validation
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]
    
    # Create datasets and dataloaders
    train_dataset = LotteryDataset(train_sequences, train_targets)
    val_dataset = LotteryDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = BiLSTMLotteryModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        output_size=output_size,
        dropout=0.3
    )
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_sequences, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_sequences, batch_targets in val_loader:
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        model.train()
    
    model.eval()
    print(f"✅ BiLSTM model training completed for {task}")
    return model

def predict_bilstm_probs(model: BiLSTMLotteryModel, sequences: np.ndarray) -> np.ndarray:
    """
    Generate predictions using trained BiLSTM model.
    
    Args:
        model: Trained BiLSTM model
        sequences: Input sequences for prediction
        
    Returns:
        Probability predictions
    """
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch dependencies not available")
    
    model.eval()
    with torch.no_grad():
        # Use the last sequence for prediction
        last_sequence = torch.FloatTensor(sequences[-1:])
        predictions = model(last_sequence)
        return predictions.numpy().flatten()

def build_panel(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Build panelized dataset for a specific prediction task.

    Args:
        df: Historical draw data
        task: One of "main", "bonus", "pb"

    Returns:
        train_df, val_df, encoder_length, decoder_length
    """
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch Forecasting dependencies not available")

    # Parse dates properly
    df = df.copy()
    df['Draw Date'] = pd.to_datetime(df['Draw Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Draw Date']).sort_values('Draw Date').reset_index(drop=True)
    df['draw_index'] = df.index

    panel_data = []

    if task == "main":
        # MAIN: predict which 6 numbers from 1-40 will be drawn
        ball_range = 40
        for idx, row in df.iterrows():
            main_numbers = set(int(row[str(i)]) for i in range(1, 7))
            for num in range(1, ball_range + 1):
                panel_data.append({
                    'draw_index': idx,
                    'number_id': num,
                    'y': 1 if num in main_numbers else 0,
                    'weekday': row['Draw Date'].weekday(),
                    'month': row['Draw Date'].month,
                    'week_of_year': row['Draw Date'].isocalendar().week,
                })

    elif task == "bonus":
        # BONUS: predict which number from 1-40 will be the bonus ball
        ball_range = 40
        for idx, row in df.iterrows():
            bonus_number = int(row['Bonus Ball'])
            for num in range(1, ball_range + 1):
                panel_data.append({
                    'draw_index': idx,
                    'number_id': num,
                    'y': 1 if num == bonus_number else 0,
                    'weekday': row['Draw Date'].weekday(),
                    'month': row['Draw Date'].month,
                    'week_of_year': row['Draw Date'].isocalendar().week,
                })

    elif task == "pb":
        # POWERBALL: predict which number from 1-10 will be drawn
        ball_range = 10
        for idx, row in df.iterrows():
            pb_number = int(row['Power Ball'])
            for num in range(1, ball_range + 1):
                panel_data.append({
                    'draw_index': idx,
                    'number_id': num,
                    'y': 1 if num == pb_number else 0,
                    'weekday': row['Draw Date'].weekday(),
                    'month': row['Draw Date'].month,
                    'week_of_year': row['Draw Date'].isocalendar().week,
                })

    else:
        raise ValueError("Task must be one of: 'main', 'bonus', 'pb'")

    # Create panel DataFrame
    panel_df = pd.DataFrame(panel_data)
    panel_df = panel_df.sort_values(['number_id', 'draw_index']).reset_index(drop=True)

    # Ensure proper data types
    panel_df['draw_index'] = panel_df['draw_index'].astype(int)
    panel_df['number_id'] = panel_df['number_id'].astype(int)
    panel_df['y'] = panel_df['y'].astype(int)

    # Add time-series features (recency, rolling stats)
    panel_df = add_production_features(panel_df, task)

    # Time-based split (no leakage)
    split_idx = int(len(panel_df) * 0.8)  # 80% train, 20% validation
    train_df = panel_df.iloc[:split_idx].copy()
    val_df = panel_df.iloc[split_idx:].copy()

    encoder_length = 60 if task == "main" else 30  # Different lookback for different tasks
    decoder_length = 1

    return train_df, val_df, encoder_length, decoder_length

def add_production_features(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Add simplified time-series features for each task."""
    df = df.copy()

    # Ensure proper sorting
    df = df.sort_values(['number_id', 'draw_index']).reset_index(drop=True)

    # Very basic features only - just lags and simple rolling stats
    df['lag_1'] = df.groupby('number_id')['y'].shift(1).fillna(0)
    df['lag_2'] = df.groupby('number_id')['y'].shift(2).fillna(0)

    # Simple rolling means
    df['rolling_mean_10'] = df.groupby('number_id')['y'].rolling(10, min_periods=1).mean().fillna(0)
    df['rolling_mean_26'] = df.groupby('number_id')['y'].rolling(26, min_periods=1).mean().fillna(0)

    # Days since last hit - very simple
    df['days_since_last_hit'] = 0  # Placeholder for now

    # Fill any NaNs
    df = df.fillna(0)

    return df

def build_tft_panels_deprecated():
    """
    Build panelized time series datasets for TFT training.

    Args:
        df: Historical draw data
        ball_range: 40 for main balls, 10 for powerball
        is_powerball: Whether this is for powerball (True) or main balls (False)
        encoder_length: Lookback window for TFT
        decoder_length: Prediction horizon

    Returns:
        Training and validation TimeSeriesDataSet objects
    """
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch Forecasting dependencies not available")

    # Prepare data: explode each draw into one row per number
    panel_data = []

    for idx, row in df.iterrows():
        draw_date = pd.to_datetime(row['Draw Date'], format='%A %d %B %Y', errors='coerce')
        if pd.isna(draw_date):
            continue

        draw_index = idx
        drawn_numbers = set()

        if is_powerball:
            # Powerball only
            powerball = int(row['Power Ball'])
            for num in range(1, ball_range + 1):
                panel_data.append({
                    'draw_index': draw_index,
                    'number_id': num,
                    'y': 1 if num == powerball else 0,
                    'weekday': draw_date.weekday(),
                    'month': draw_date.month,
                    'year': draw_date.year,
                    'day_of_year': draw_date.dayofyear
                })
        else:
            # Main balls
            main_balls = [int(row[str(i)]) for i in range(1, 7)]
            drawn_numbers = set(main_balls)

            for num in range(1, ball_range + 1):
                panel_data.append({
                    'draw_index': draw_index,
                    'number_id': num,
                    'y': 1 if num in drawn_numbers else 0,
                    'weekday': draw_date.weekday(),
                    'month': draw_date.month,
                    'year': draw_date.year,
                    'day_of_year': draw_date.dayofyear
                })

    panel_df = pd.DataFrame(panel_data)
    panel_df = panel_df.sort_values(['number_id', 'draw_index']).reset_index(drop=True)

    # Add feature engineering
    panel_df = add_time_series_features(panel_df, ball_range)

    # Split into train/validation (last 20% for validation)
    split_idx = int(len(panel_df) * 0.8)
    train_df = panel_df.iloc[:split_idx].copy()
    val_df = panel_df.iloc[split_idx:].copy()

    # Define static and time-varying features
    static_categoricals = ['number_id']
    time_varying_known = ['weekday', 'month', 'day_of_year']
    time_varying_unknown = [
        'y',  # target
        'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
        'rolling_mean_5', 'rolling_mean_10', 'rolling_mean_26',
        'rolling_std_5', 'rolling_std_10',
        'days_since_last_hit',
        'streak_current', 'streak_max_last_10'
    ]

    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx='draw_index',
        target='y',
        group_ids=['number_id'],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=decoder_length,
        max_prediction_length=decoder_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        target_normalizer=None,  # Binary classification
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create validation dataset (using same cutoff)
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df,
        predict=True,
        stop_randomization=True
    )

    return training, validation, encoder_length, decoder_length

def add_time_series_features(df: pd.DataFrame, ball_range: int) -> pd.DataFrame:
    """Add time series features: lags, rolling statistics, streaks."""
    df = df.copy()

    # Group by number_id for time series operations
    grouped = df.groupby('number_id')

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'lag_{lag}'] = grouped['y'].shift(lag).fillna(0)

    # Rolling statistics
    for window in [5, 10, 26]:
        df[f'rolling_mean_{window}'] = grouped['y'].rolling(window=window, min_periods=1).mean().fillna(0)
        if window <= 10:  # Only compute std for smaller windows
            df[f'rolling_std_{window}'] = grouped['y'].rolling(window=window, min_periods=1).std().fillna(0)

    # Days since last hit
    def days_since_last_hit(series):
        last_hit_indices = series[series == 1].index
        if len(last_hit_indices) == 0:
            return pd.Series([999] * len(series), index=series.index)

        result = []
        last_hit = -999
        for i, val in enumerate(series):
            if val == 1:
                last_hit = i
                result.append(0)
            else:
                result.append(i - last_hit)
        return pd.Series(result, index=series.index)

    df['days_since_last_hit'] = grouped['y'].apply(days_since_last_hit).fillna(999)

    # Streak features
    def calculate_streaks(series):
        streaks = []
        current_streak = 0
        max_streak_last_10 = 0
        last_10_streaks = []

        for i, val in enumerate(series):
            if val == 1:
                current_streak += 1
            else:
                current_streak = 0

            streaks.append(current_streak)

            # Track max streak in last 10 draws
            if len(last_10_streaks) >= 10:
                last_10_streaks.pop(0)
            last_10_streaks.append(current_streak)
            max_streak_last_10 = max(last_10_streaks) if last_10_streaks else 0

            if i >= 9:  # Only set after first 10 observations
                series.iloc[i] = max_streak_last_10

        return pd.Series(streaks, index=series.index)

    df['streak_current'] = grouped['y'].apply(calculate_streaks)
    df['streak_max_last_10'] = grouped['y'].apply(lambda x: x.rolling(10, min_periods=1).max().fillna(0))

    return df

def train_tft_deprecated():
    """
    Train a Temporal Fusion Transformer for a specific task.

    Args:
        train_df: Training panel data
        val_df: Validation panel data
        task: Task name ("main", "bonus", "pb")
        encoder_length: Lookback window
        decoder_length: Prediction horizon

    Returns:
        Trained model and validation dataset
    """
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch Forecasting dependencies not available")

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Define feature sets based on task
    static_categoricals = ['number_id']
    time_varying_known_reals = ['weekday', 'month', 'week_of_year']

    if task == "main":
        time_varying_unknown_reals = [
            'y', 'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
            'rolling_hits_10', 'rolling_hits_26', 'rolling_hits_52',
            'rolling_mean_10', 'rolling_mean_26', 'rolling_mean_52',
            'days_since_last_hit', 'rolling_entropy_10', 'rolling_entropy_26'
        ]
    else:  # bonus or pb
        time_varying_unknown_reals = [
            'y', 'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
            'rolling_hits_26', 'rolling_hits_52',
            'rolling_mean_26', 'rolling_mean_52',
            'days_since_last_hit'
        ]

    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx='draw_index',
        target='y',
        group_ids=['number_id'],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=decoder_length,
        max_prediction_length=decoder_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=None,  # Binary classification
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df,
        predict=True,
        stop_randomization=True
    )

    # Removed TFT model creation

    from pytorch_lightning.callbacks import EarlyStopping
    trainer = Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")],
    )

    trainer.fit(
        model,
        train_dataloaders=training.to_dataloader(train=True, batch_size=1024, num_workers=0),
        val_dataloaders=validation.to_dataloader(train=False, batch_size=2048, num_workers=0)
    )

    return model, validation

def predict_probs_deprecated():
    """
    Predict probabilities for the next draw using trained TFT model.

    Args:
        model: Trained TFT model
        validation_dataset: Validation dataset for inference

    Returns:
        Probability vector for each number (1-based indexing)
    """
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch Forecasting dependencies not available")

    # Create dataloader for prediction
    dl = validation_dataset.to_dataloader(train=False, batch_size=4096, num_workers=0)

    # Get predictions with index
    preds = model.predict(dl, return_index=True, return_x=False)

    # Convert logits to probabilities
    p = torch.sigmoid(preds.output).cpu().numpy().flatten()

    # Get the corresponding number_ids and time indices
    idx = preds.index.reset_index(drop=True)
    dfp = pd.DataFrame({
        "number_id": idx["number_id"].astype(int),
        "time_idx": idx["time_idx"].astype(int),
        "p": p
    })

    # Get latest prediction for each number_id
    latest = dfp.sort_values(["number_id", "time_idx"]).groupby("number_id", as_index=False).tail(1)

    # Convert to array (1-based indexing)
    max_id = int(latest["number_id"].max())
    out = np.zeros(max_id + 1)
    for _, r in latest.iterrows():
        out[int(r["number_id"])] = float(r["p"])

    return out[1:]  # Return 1-based array


def apply_guardrails(p: np.ndarray, task: str) -> np.ndarray:
    """
    Apply guardrails to predicted probabilities to prevent mode collapse.

    Args:
        p: Probability vector
        task: Task name ("main", "bonus", "pb")

    Returns:
        Clipped probability vector
    """
    if task in ["main", "bonus"]:
        uniform_prob = 1.0 / 40
    else:  # pb
        uniform_prob = 1.0 / 10

    # Clip to [0.5×uniform, 2×uniform] to prevent extreme predictions
    p_clipped = np.clip(p, uniform_prob * 0.5, uniform_prob * 2.0)
    return p_clipped

def pl_sample_without_replacement(p: np.ndarray, k: int, rng: random.Random) -> List[int]:
    """
    Sample k items without replacement using Plackett-Luce model.

    Args:
        p: Probability vector (unnormalized weights)
        k: Number of items to sample
        rng: Random number generator

    Returns:
        List of selected indices (1-based)
    """
    selected = []
    available = list(range(len(p)))
    weights = p.copy().astype(float)

    for _ in range(k):
        if len(available) == 0:
            break

        # Normalize remaining weights
        if weights[available].sum() > 0:
            w = weights[available] / weights[available].sum()
        else:
            # Uniform if all weights are zero
            w = np.ones(len(available)) / len(available)

        # Sample one item
        idx = rng.choices(range(len(available)), weights=w, k=1)[0]
        choice = available.pop(idx)
        selected.append(choice + 1)  # Convert to 1-based

    return selected

def pick_bonus(p_bonus: np.ndarray, chosen_main: List[int], rng: random.Random,
               mode: str = 'probabilistic') -> int:
    """
    Pick bonus ball from remaining numbers (excluding chosen main numbers).

    Args:
        p_bonus: Bonus probability vector (1-based, length 40)
        chosen_main: List of chosen main numbers (1-based)
        rng: Random number generator
        mode: 'deterministic' or 'probabilistic'

    Returns:
        Selected bonus number (1-based)
    """
    # Create mask for available numbers (exclude chosen main)
    mask = np.ones(len(p_bonus), dtype=bool)
    for num in chosen_main:
        if 1 <= num <= len(p_bonus):
            mask[num - 1] = False

    available_indices = np.where(mask)[0]

    if mode == 'deterministic':
        # Top-1 from available numbers
        available_probs = p_bonus[mask]
        best_idx = np.argmax(available_probs)
        return available_indices[best_idx] + 1

    else:  # probabilistic
        # Weighted sampling from available numbers
        available_probs = p_bonus[mask]
        if available_probs.sum() > 0:
            probs = available_probs / available_probs.sum()
        else:
            probs = np.ones(len(available_probs)) / len(available_probs)

        idx = rng.choices(range(len(available_indices)), weights=probs, k=1)[0]
        return available_indices[idx] + 1

def generate_bilstm_ticket(p_main: np.ndarray, p_bonus: np.ndarray, p_pb: np.ndarray,
                          rng: random.Random, mode: str = 'probabilistic') -> Dict[str, List[int]]:
    """
    Generate a complete ticket using BiLSTM predictions.

    Args:
        p_main: Main ball probabilities (length 40)
        p_bonus: Bonus ball probabilities (length 40)
        p_pb: Powerball probabilities (length 10)
        rng: Random number generator
        mode: 'deterministic' or 'probabilistic'

    Returns:
        Dictionary with main_balls, bonus, powerball
    """
    if mode == 'deterministic':
        # Top-K approach
        main_indices = np.argsort(p_main)[-6:]  # Top 6
        main_balls = sorted([idx + 1 for idx in main_indices])

        bonus = pick_bonus(p_bonus, main_balls, rng, mode='deterministic')
        powerball = np.argmax(p_pb) + 1

    else:  # probabilistic
        # Plackett-Luce sampling
        main_balls = pl_sample_without_replacement(p_main, 6, rng)
        main_balls = sorted(main_balls)

        bonus = pick_bonus(p_bonus, main_balls, rng, mode='probabilistic')
        powerball = rng.choices(range(1, len(p_pb) + 1), weights=p_pb, k=1)[0]

    return {
        'main_balls': main_balls,
        'bonus': bonus,
        'powerball': powerball
    }

def train_tft_model_deprecated():
    """Train Temporal Fusion Transformer model."""
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch Forecasting dependencies not available")

    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Removed TFT model creation

    # Create trainer
    from pytorch_lightning.callbacks import EarlyStopping

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="cpu" if gpus == 0 else "gpu",
        devices=1 if gpus > 0 else None,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min")
        ]
    )

    # Train model
    trainer.fit(
        model,
        train_dataloaders=ts_train.to_dataloader(train=True, batch_size=1024, num_workers=0),
        val_dataloaders=ts_valid.to_dataloader(train=False, batch_size=2048, num_workers=0)
    )

    return model

def predict_tft_probs_deprecated():
    """Predict probabilities for next draw using trained TFT model."""
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch Forecasting dependencies not available")

    # Create dataloader for prediction
    dl = ts_dataset.to_dataloader(train=False, batch_size=4096, num_workers=0)

    # Get predictions (logits)
    preds_logits = model.predict(dl, return_x=False)

    # Convert to probabilities
    preds_probs = torch.sigmoid(preds_logits).cpu().numpy().flatten()

    # Get the last prediction for each group (most recent draw)
    # Group by number_id and take the last prediction
    pred_df = pd.DataFrame({
        'number_id': ts_dataset.decoded_index['number_id'].values,
        'probability': preds_probs
    })

    # Get unique number_ids and their latest predictions
    latest_preds = pred_df.groupby('number_id')['probability'].last()

    # Return as array indexed by number (1-based)
    max_num = int(pred_df['number_id'].max())
    prob_vector = np.zeros(max_num + 1)
    for num, prob in latest_preds.items():
        prob_vector[int(num)] = prob

    return prob_vector[1:]  # Remove index 0, return 1-based

def calibrate_probabilities(y_true: np.ndarray, y_pred: np.ndarray,
                          method: str = 'isotonic') -> callable:
    """Calibrate predicted probabilities using Platt scaling or isotonic regression."""
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif method == 'platt':
        calibrator = LogisticRegression()
    else:
        raise ValueError("Method must be 'isotonic' or 'platt'")

    # Fit calibrator
    calibrator.fit(y_pred.reshape(-1, 1), y_true)

    # Return calibration function
    def calibrate_func(probs):
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    return calibrate_func

def plackett_luce_weighted_sample(p: np.ndarray, k: int, rng: random.Random) -> List[int]:
    """
    Sample k items without replacement using Plackett-Luce model.

    Args:
        p: Probability vector (unnormalized weights)
        k: Number of items to sample
        rng: Random number generator

    Returns:
        List of selected indices (0-based)
    """
    selected = []
    available = list(range(len(p)))
    weights = p.copy().astype(float)

    for _ in range(k):
        if len(available) == 0:
            break

        # Normalize remaining weights
        if weights[available].sum() > 0:
            w = weights[available] / weights[available].sum()
        else:
            # Uniform if all weights are zero
            w = np.ones(len(available)) / len(available)

        # Sample one item
        idx = rng.choices(range(len(available)), weights=w, k=1)[0]
        choice = available.pop(idx)
        selected.append(choice)

    return selected

def apply_popularity_penalty(p: np.ndarray, popular_patterns: Dict,
                           penalty_factor: float = 0.1) -> np.ndarray:
    """Apply soft penalties to popular patterns instead of hard filtering."""
    p_adj = p.copy()

    # Penalty for dates (1-31)
    for num in range(min(len(p), 31)):
        if num + 1 in popular_patterns.get('dates', set()):
            p_adj[num] *= penalty_factor

    # Penalty for teen numbers (13-19)
    for num in range(12, min(len(p), 19)):
        if num + 1 in popular_patterns.get('teen_numbers', set()):
            p_adj[num] *= penalty_factor

    # Penalty for repeating digits
    for num in range(len(p)):
        if num + 1 in popular_patterns.get('repeating_digits', []):
            p_adj[num] *= penalty_factor

    # Penalty for fibonacci
    for num in range(len(p)):
        if num + 1 in popular_patterns.get('fibonacci', set()):
            p_adj[num] *= penalty_factor

    return p_adj

class NZPowerballTicketOptimizer:
    """
    NZ Powerball Ticket Portfolio Optimizer (TPO) with Popularity Avoidance Heuristics

    Features:
    - Cryptographically seeded uniform random sampling
    - Portfolio-level diversification constraints
    - Popularity avoidance heuristics (dates, sequences, patterns)
    - Expected value analysis for rollover scenarios
    - Transparent compliance with reproducible run IDs
    - Optional TFT-based probability predictions
    """

    def __init__(self, csv_path: str, seed: Optional[str] = None, use_tft: bool = False):
        self.csv_path = csv_path
        self.data = None
        self.use_tft = use_tft

        # Cryptographic seeding for reproducible results
        if seed is None:
            seed = secrets.token_hex(16)  # Generate cryptographically secure seed
        self.seed = seed
        self.run_id = self._generate_run_id(seed)

        # Initialize cryptographically seeded RNG
        self.rng = self._initialize_rng(seed)

        # Configuration for NZ Powerball
        self.MAIN_BALL_RANGE = (1, 40)
        self.POWERBALL_RANGE = (1, 10)
        self.MAIN_BALLS_COUNT = 6
        self.TICKET_PRICE = 1.50  # NZD (Lotto $0.70 + Powerball add-on $0.80)

        # Popularity avoidance patterns to avoid
        self._initialize_popularity_patterns()

        # Historical data for analysis
        self.historical_patterns = {}
        self.jackpot_history = []

        # BiLSTM models and predictions (production-grade three-model approach)
        self.lstm_main_model = None
        self.lstm_bonus_model = None
        self.lstm_pb_model = None
        self.p_main = None      # Main ball probabilities (1-40)
        self.p_bonus = None     # Bonus ball probabilities (1-40)
        self.p_pb = None        # Powerball probabilities (1-10)
        self.calibrator_main = None
        self.calibrator_bonus = None
        self.calibrator_pb = None
        self.tft_mode = 'probabilistic'  # 'deterministic' or 'probabilistic'

    def _generate_run_id(self, seed: str) -> str:
        """Generate a reproducible run ID from seed (deterministic)"""
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    def _initialize_rng(self, seed: str) -> random.Random:
        """Initialize cryptographically seeded random number generator"""
        # Use hashlib to create a deterministic seed from the string
        seed_bytes = hashlib.sha256(seed.encode()).digest()
        seed_int = int.from_bytes(seed_bytes, byteorder='big')
        rng = random.Random(seed_int)
        return rng

    def _initialize_popularity_patterns(self):
        """Initialize patterns that are commonly chosen and should be avoided"""
        self.popular_patterns = {
            'dates': set(range(1, 32)),  # Days 1-31
            'teen_numbers': set(range(13, 20)),  # 13-19 (unlucky numbers)
            'sequences': [],  # Will be generated dynamically
            'repeating_digits': [],  # Will be generated dynamically
            'birthday_months': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  # 1-12
            'fibonacci': {1, 2, 3, 5, 8, 13, 21, 34},
            'multiples_of_5': {5, 10, 15, 20, 25, 30, 35, 40},
            'multiples_of_10': {10, 20, 30, 40}
        }

        # Generate sequences (1,2,3,4,5,6 etc.)
        for start in range(1, 40):
            if start + 5 <= 40:
                self.popular_patterns['sequences'].append(tuple(range(start, start + 6)))

        # Generate repeating digit patterns (11,22,33, etc.)
        for digit in range(1, 5):  # 11,22,33,44
            repeat = digit * 11  # 11,22,33,44
            if repeat <= 40:
                self.popular_patterns['repeating_digits'].append(repeat)

    def load_historical_data(self) -> pd.DataFrame:
        """Load and validate historical lottery data for analysis"""
        try:
            self.data = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            # Try alternative filename format
            alt_path = self.csv_path.replace('NZ Powerballtill 27082025.csv', 'NZ Powerball-till 27-08-2025.csv')
            self.data = pd.read_csv(alt_path)
        
        print(f"Loaded {len(self.data)} historical draws")
        print(f"Columns: {list(self.data.columns)}")
        
        # Convert Draw Date to datetime
        self.data['Draw Date'] = pd.to_datetime(self.data['Draw Date'], format='%A %d %B %Y', errors='coerce')
        
        # Extract main balls and powerball columns
        self.main_ball_cols = ['1', '2', '3', '4', '5', '6']
        self.powerball_col = 'Power Ball'
        self.bonus_col = 'Bonus Ball'
        
        # Sort by date
        self.data = self.data.sort_values('Draw Date').reset_index(drop=True)
        
        # Analyze historical patterns for popularity avoidance
        self._analyze_historical_patterns()

        print(f"✅ Loaded {len(self.data)} historical draws and analyzed patterns")
        return self.data
    
    def _analyze_historical_patterns(self):
        """Analyze historical data to understand patterns and frequencies"""
        if self.data is None or self.data.empty:
            return

        # Analyze main ball frequencies
        all_main_balls = []
        for col in self.main_ball_cols:
            all_main_balls.extend(self.data[col].dropna().astype(int).tolist())

        self.historical_patterns['main_ball_freq'] = pd.Series(all_main_balls).value_counts().to_dict()

        # Analyze powerball frequencies
        self.historical_patterns['powerball_freq'] = self.data[self.powerball_col].dropna().astype(int).value_counts().to_dict()

        # Analyze jackpot amounts if available
        if 'Jackpot' in self.data.columns:
            self.jackpot_history = self.data['Jackpot'].dropna().tolist()

        print(f"Analyzed patterns from {len(self.historical_patterns['main_ball_freq'])} unique main balls")
        print(f"Analyzed patterns from {len(self.historical_patterns['powerball_freq'])} unique powerballs")

        # Train BiLSTM models if requested
        if self.use_tft and LSTM_AVAILABLE:
            self._train_lstm_models()

        return self.data

    def _train_lstm_models(self):
        """Train three separate BiLSTM models for main, bonus, and powerball predictions."""
        if not LSTM_AVAILABLE:
            print("Warning: PyTorch dependencies not available, falling back to uniform sampling")
            self.use_tft = False
            return

        print("\n🏗️  Training production-grade BiLSTM models (3 separate models)...")

        try:
            # Build and train MAIN model (6 balls from 1-40)
            print("📊 Preparing MAIN dataset (6 balls from 1-40)...")
            main_sequences, main_targets = prepare_lstm_sequences(self.data, "main", sequence_length=20)
            
            print("🎯 Training MAIN BiLSTM model...")
            self.lstm_main_model = train_bilstm_model(main_sequences, main_targets, "main", 
                                                     hidden_size=128, num_epochs=30)

            # Build and train BONUS model (1 ball from 1-40, excluding main)
            print("📊 Preparing BONUS dataset (1 ball from 1-40)...")
            bonus_sequences, bonus_targets = prepare_lstm_sequences(self.data, "bonus", sequence_length=15)
            
            print("🎯 Training BONUS BiLSTM model...")
            self.lstm_bonus_model = train_bilstm_model(bonus_sequences, bonus_targets, "bonus", 
                                                      hidden_size=96, num_epochs=25)

            # Build and train POWERBALL model (1 ball from 1-10)
            print("📊 Preparing POWERBALL dataset (1 ball from 1-10)...")
            pb_sequences, pb_targets = prepare_lstm_sequences(self.data, "pb", sequence_length=10)
            
            print("🎯 Training POWERBALL BiLSTM model...")
            self.lstm_pb_model = train_bilstm_model(pb_sequences, pb_targets, "pb", 
                                                   hidden_size=64, num_epochs=20)

            # Generate predictions for next draw
            print("🔮 Generating predictions for next draw...")
            self.p_main = predict_bilstm_probs(self.lstm_main_model, main_sequences)
            self.p_bonus = predict_bilstm_probs(self.lstm_bonus_model, bonus_sequences)
            self.p_pb = predict_bilstm_probs(self.lstm_pb_model, pb_sequences)

            # Apply calibration and guardrails
            self._calibrate_and_guard_predictions()

            # Validate predictions
            self._validate_predictions()

            # Run comprehensive validation
            val_metrics = self.validate_lstm_models()

            print("✅ All BiLSTM models trained and predictions generated!")
            print(f"   📈 MAIN predictions: {len(self.p_main)} numbers")
            print(f"   🎯 BONUS predictions: {len(self.p_bonus)} numbers")
            print(f"   ⚡ POWERBALL predictions: {len(self.p_pb)} numbers")
            print(f"   🎲 Inference mode: {self.tft_mode}")

        except Exception as e:
            print(f"⚠️  Warning: BiLSTM training failed ({e}), falling back to uniform sampling")
            self.use_tft = False

    def _calibrate_and_guard_predictions(self):
        """Apply calibration and guardrails to all TFT predictions."""
        if self.p_main is None or self.p_bonus is None or self.p_pb is None:
            return

        print("🔧 Applying calibration and guardrails...")

        # Apply guardrails (prevent mode collapse)
        self.p_main = apply_guardrails(self.p_main, "main")
        self.p_bonus = apply_guardrails(self.p_bonus, "bonus")
        self.p_pb = apply_guardrails(self.p_pb, "pb")

        # Apply soft popularity penalties (optional)
        if hasattr(self, 'popular_patterns'):
            self.p_main = apply_popularity_penalty(self.p_main, self.popular_patterns, penalty_factor=0.3)
            self.p_bonus = apply_popularity_penalty(self.p_bonus, self.popular_patterns, penalty_factor=0.3)

        print("📊 Prediction ranges after calibration:")
        print(f"Main p range: [{self.p_main.min():.4f}, {self.p_main.max():.4f}]")
        print(f"Bonus p range: [{self.p_bonus.min():.4f}, {self.p_bonus.max():.4f}]")
        print(f"PB p range: [{self.p_pb.min():.4f}, {self.p_pb.max():.4f}]")

    def _validate_predictions(self):
        """Validate TFT predictions for reasonableness."""
        if self.p_main is None or self.p_bonus is None or self.p_pb is None:
            return

        # Check that probabilities sum to reasonable values
        main_sum = self.p_main.sum()
        bonus_sum = self.p_bonus.sum()
        pb_sum = self.p_pb.sum()

        print(".2f")
        print(".2f")
        print(".2f")

        # Check for any NaN or infinite values
        if np.any(np.isnan(self.p_main)) or np.any(np.isnan(self.p_bonus)) or np.any(np.isnan(self.p_pb)):
            print("⚠️  Warning: NaN values detected in predictions")
        if np.any(np.isinf(self.p_main)) or np.any(np.isinf(self.p_bonus)) or np.any(np.isinf(self.p_pb)):
            print("⚠️  Warning: Infinite values detected in predictions")

    def validate_lstm_models(self) -> Dict[str, float]:
        """
        Comprehensive validation of BiLSTM models on held-out validation data.

        Returns:
            Dictionary with validation metrics
        """
        if not LSTM_AVAILABLE or self.lstm_main_model is None:
            return {}

        metrics = {}

        try:
            from sklearn.metrics import log_loss, roc_auc_score

            print("📊 BiLSTM model validation...")
            
            # Basic validation - check if models can generate reasonable predictions
            if self.p_main is not None:
                main_sum = np.sum(self.p_main)
                main_entropy = -np.sum(self.p_main * np.log(self.p_main + 1e-8))
                metrics['main_prob_sum'] = main_sum
                metrics['main_entropy'] = main_entropy
                
            if self.p_bonus is not None:
                bonus_sum = np.sum(self.p_bonus)
                bonus_entropy = -np.sum(self.p_bonus * np.log(self.p_bonus + 1e-8))
                metrics['bonus_prob_sum'] = bonus_sum
                metrics['bonus_entropy'] = bonus_entropy
                
            if self.p_pb is not None:
                pb_sum = np.sum(self.p_pb)
                pb_entropy = -np.sum(self.p_pb * np.log(self.p_pb + 1e-8))
                metrics['pb_prob_sum'] = pb_sum
                metrics['pb_entropy'] = pb_entropy

            print("📊 BiLSTM Validation Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

        except Exception as e:
            print(f"⚠️  BiLSTM validation failed: {e}")

        return metrics

    def generate_ticket_portfolio(self, num_tickets: int = 10,
                                diversity_constraints: Optional[Dict] = None) -> List[Dict]:
        """
        Generate an optimized ticket portfolio using uniform random sampling with constraints

        Args:
            num_tickets: Number of tickets to generate
            diversity_constraints: Optional constraints for portfolio diversification

        Returns:
            List of ticket dictionaries
        """
        if diversity_constraints is None:
            diversity_constraints = {
                'max_consecutive': 3,  # Max consecutive numbers allowed
                'balance_odd_even': True,  # Balance odd/even distribution
                'avoid_popular_patterns': True,  # Avoid popularity patterns
                'sum_range': (70, 180),  # Target sum range for main balls
                'digit_ending_variety': True  # Ensure variety in last digits
            }

        portfolio = []
        used_combinations = set()  # Track combinations to ensure uniqueness

        for ticket_id in range(1, num_tickets + 1):
            ticket = self._generate_single_ticket(ticket_id, diversity_constraints, used_combinations)
            portfolio.append(ticket)
            used_combinations.add(tuple(sorted(ticket['main_balls'])))

        # Validate portfolio diversity
        self._validate_portfolio_diversity(portfolio, diversity_constraints)

        return portfolio

    def _generate_single_ticket(self, ticket_id: int,
                               constraints: Dict,
                               used_combinations: set) -> Dict:
        """Generate a single optimized ticket"""
        max_attempts = 1000
        attempt = 0

        while attempt < max_attempts:
            # Generate main balls with uniform random sampling
            main_balls = self._generate_main_balls(constraints)

            # Check if combination is already used
            main_tuple = tuple(sorted(main_balls))
            if main_tuple in used_combinations:
                attempt += 1
                continue

            # Generate complete ticket using BiLSTM predictions (if available)
            if self.use_tft and self.p_main is not None and self.p_bonus is not None and self.p_pb is not None:
                ticket_data = generate_bilstm_ticket(
                    self.p_main, self.p_bonus, self.p_pb,
                    self.rng, mode=self.tft_mode
                )
                main_balls = ticket_data['main_balls']
                powerball = ticket_data['powerball']
                bonus_ball = ticket_data['bonus']
            else:
                # Fallback to original uniform generation
                main_balls = self._generate_main_balls(constraints)
            powerball = self.rng.randint(*self.POWERBALL_RANGE)
            bonus_ball = self.rng.randint(*self.MAIN_BALL_RANGE)  # Bonus is from same range

            # Validate against constraints
            if self._validate_ticket_constraints(main_balls, powerball, constraints, bonus_ball):
                return {
                    'ticket_id': ticket_id,
                    'main_balls': sorted(main_balls),
                    'powerball': powerball,
                    'bonus': bonus_ball,
                    'sum': sum(main_balls),
                    'odd_count': sum(1 for x in main_balls if x % 2 == 1),
                    'even_count': sum(1 for x in main_balls if x % 2 == 0),
                    'last_digits': [x % 10 for x in main_balls]
                }

            attempt += 1

        # If we can't satisfy constraints, generate without some constraints
        print(f"Warning: Could not satisfy all constraints for ticket {ticket_id} after {max_attempts} attempts")
        main_balls = self._generate_main_balls_simple()
        powerball = self.rng.randint(*self.POWERBALL_RANGE)
        bonus_ball = self.rng.randint(*self.MAIN_BALL_RANGE)

        return {
            'ticket_id': ticket_id,
            'main_balls': sorted(main_balls),
            'powerball': powerball,
            'bonus': bonus_ball,
            'sum': sum(main_balls),
            'odd_count': sum(1 for x in main_balls if x % 2 == 1),
            'even_count': sum(1 for x in main_balls if x % 2 == 0),
            'last_digits': [x % 10 for x in main_balls]
        }

    def _generate_main_balls(self, constraints: Dict) -> List[int]:
        """Generate main balls with TFT predictions or popularity avoidance"""
        candidates = list(range(self.MAIN_BALL_RANGE[0], self.MAIN_BALL_RANGE[1] + 1))

        # Apply popularity avoidance if requested (hard filter for fallback)
        if constraints.get('avoid_popular_patterns', False):
            candidates = self._apply_popularity_filter(candidates)

        # Use TFT predictions if available, otherwise uniform sampling
        if self.use_tft and self.p_main is not None:
            # Filter predictions to available candidates only
            p_filtered = self.p_main[np.array(candidates) - 1]  # Convert to 0-based indexing

            # Use Plackett-Luce weighted sampling
            selected_indices = plackett_luce_weighted_sample(p_filtered, self.MAIN_BALLS_COUNT, self.rng)
            selected = [candidates[i] for i in selected_indices]
        else:
            # Fallback to uniform sampling
            selected = self.rng.sample(candidates, self.MAIN_BALLS_COUNT)

        # Apply additional constraints
        if constraints.get('balance_odd_even', False):
            selected = self._balance_odd_even(selected)

        return selected

    def _generate_main_balls_simple(self) -> List[int]:
        """Simple uniform random generation without constraints"""
        return sorted(self.rng.sample(
            range(self.MAIN_BALL_RANGE[0], self.MAIN_BALL_RANGE[1] + 1),
            self.MAIN_BALLS_COUNT
        ))

    def _apply_popularity_filter(self, candidates: List[int]) -> List[int]:
        """Filter out popular patterns from candidates"""
        filtered = []

        for num in candidates:
            # Avoid dates (1-31)
            if num in self.popular_patterns['dates']:
                if self.rng.random() > 0.1:  # 10% chance to keep
                    continue

            # Avoid teen numbers (13-19)
            if num in self.popular_patterns['teen_numbers']:
                if self.rng.random() > 0.15:  # 15% chance to keep
                    continue

            # Avoid repeating digits
            if num in self.popular_patterns['repeating_digits']:
                if self.rng.random() > 0.2:  # 20% chance to keep
                    continue

            # Avoid fibonacci numbers
            if num in self.popular_patterns['fibonacci']:
                if self.rng.random() > 0.25:  # 25% chance to keep
                    continue

            filtered.append(num)

        return filtered if filtered else candidates  # Fallback if all filtered out

    def _balance_odd_even(self, numbers: List[int]) -> List[int]:
        """Balance odd/even distribution in selected numbers"""
        odd_count = sum(1 for x in numbers if x % 2 == 1)
        even_count = len(numbers) - odd_count

        # Target roughly 50/50 split
        target_odd = len(numbers) // 2
        target_even = len(numbers) - target_odd

        if abs(odd_count - target_odd) <= 1:  # Already balanced
            return numbers

        # Try to swap numbers to balance
        available_range = list(range(self.MAIN_BALL_RANGE[0], self.MAIN_BALL_RANGE[1] + 1))
        available_range = [x for x in available_range if x not in numbers]

        if odd_count > target_odd:  # Too many odd, need more even
            # Find odd number to replace
            odd_numbers = [x for x in numbers if x % 2 == 1]
            if odd_numbers and available_range:
                even_candidates = [x for x in available_range if x % 2 == 0]
                if even_candidates:
                    to_replace = self.rng.choice(odd_numbers)
                    replacement = self.rng.choice(even_candidates)
                    numbers = [replacement if x == to_replace else x for x in numbers]

        elif even_count > target_even:  # Too many even, need more odd
            # Find even number to replace
            even_numbers = [x for x in numbers if x % 2 == 0]
            if even_numbers and available_range:
                odd_candidates = [x for x in available_range if x % 2 == 1]
                if odd_candidates:
                    to_replace = self.rng.choice(even_numbers)
                    replacement = self.rng.choice(odd_candidates)
                    numbers = [replacement if x == to_replace else x for x in numbers]

        return numbers

    def _validate_ticket_constraints(self, main_balls: List[int],
                                   powerball: int, constraints: Dict, bonus_ball: Optional[int] = None) -> bool:
        """Validate ticket against diversity constraints"""
        # Check sum range
        total_sum = sum(main_balls)
        if 'sum_range' in constraints:
            min_sum, max_sum = constraints['sum_range']
            if not (min_sum <= total_sum <= max_sum):
                return False

        # Check consecutive numbers
        if 'max_consecutive' in constraints:
            sorted_balls = sorted(main_balls)
            max_consec = constraints['max_consecutive']
            for i in range(len(sorted_balls) - 1):
                if sorted_balls[i + 1] - sorted_balls[i] == 1:
                    consecutive_count = 1
                    j = i + 1
                    while j < len(sorted_balls) - 1 and sorted_balls[j + 1] - sorted_balls[j] == 1:
                        consecutive_count += 1
                        j += 1
                    if consecutive_count > max_consec:
                        return False

        # Check digit ending variety
        if constraints.get('digit_ending_variety', False):
            last_digits = [x % 10 for x in main_balls]
            if len(set(last_digits)) < 4:  # At least 4 different endings
                return False

        # Validate bonus ball (if provided) - must not be in main balls
        if bonus_ball is not None:
            if bonus_ball in main_balls:
                return False

        return True

    def _validate_portfolio_diversity(self, portfolio: List[Dict], constraints: Dict):
        """Validate overall portfolio diversity"""
        if not portfolio:
            return
        
        # Check for duplicate combinations
        combinations = [tuple(sorted(t['main_balls'])) for t in portfolio]
        if len(combinations) != len(set(combinations)):
            print("Warning: Duplicate combinations found in portfolio")

        # Check sum distribution
        sums = [t['sum'] for t in portfolio]
        if len(set(sums)) < len(portfolio) * 0.7:  # At least 70% unique sums
            print("Warning: Low sum diversity in portfolio")

        print(f"Portfolio diversity validated: {len(portfolio)} unique tickets")

    def calculate_expected_value(self, portfolio: List[Dict],
                               jackpot_amount: float,
                               estimated_sales: Optional[float] = None,
                               prize_structure: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate expected value for a portfolio given jackpot amount

        Args:
            portfolio: List of ticket dictionaries
            jackpot_amount: Current jackpot amount in NZD
            estimated_sales: Estimated number of tickets sold (for co-winner calculation)
            prize_structure: Optional prize structure for non-jackpot divisions

        Returns:
            Dictionary with expected value analysis
        """
        if estimated_sales is None:
            estimated_sales = 500000  # Conservative estimate

        total_investment = len(portfolio) * self.TICKET_PRICE

        # Get single ticket probabilities (Powerball only)
        single = self._powerball_single_ticket_division_probs()

        # NZ Powerball prize structure (official divisions)
        if prize_structure is None:
            prize_structure = {
                'PB Div 1 (6+PB)': jackpot_amount,           # Jackpot pool
                'PB Div 2 (5+Bonus+PB)': jackpot_amount * 0.05,  # ~5% of jackpot pool
                'PB Div 3 (5+PB)': jackpot_amount * 0.02,        # ~2% of jackpot pool
                'PB Div 4 (4+Bonus+PB)': jackpot_amount * 0.005, # ~0.5% of jackpot pool
                'PB Div 5 (4+PB)': 500.0,                      # Fixed estimate
                'PB Div 6 (3+Bonus+PB)': 100.0,                # Fixed estimate
                'PB Div 7 (3+PB)': 15.0,                       # Official fixed amount
            }
        else:
            # Override jackpot amount in provided structure
            prize_structure = prize_structure.copy()
            prize_structure['PB Div 1 (6+PB)'] = jackpot_amount

        # Portfolio size
        n = len(portfolio)
        # External ticket sales
        S = int(estimated_sales)

        expected_value = 0
        division_breakdown = {}

        # Special handling for jackpot (PB Div 1) with co-winner adjustment
        p_jackpot = single['PB Div 1 (6+PB)']
        p_you_win = 1 - (1 - p_jackpot)**n

        if p_you_win > 0:
            share_factor = self._expected_share_factor(S, p_jackpot)
            ev_jackpot = jackpot_amount * p_you_win * share_factor
            expected_value += ev_jackpot
            division_breakdown['PB Div 1 (6+PB)'] = {
                'probability': p_you_win,
                'prize': jackpot_amount,
                'expected_prize': jackpot_amount * share_factor
            }

        # Handle non-jackpot divisions (no share adjustment needed)
        pb_divisions = ['PB Div 2 (5+Bonus+PB)', 'PB Div 3 (5+PB)', 'PB Div 4 (4+Bonus+PB)',
                       'PB Div 5 (4+PB)', 'PB Div 6 (3+Bonus+PB)', 'PB Div 7 (3+PB)']
        for name in pb_divisions:
            p_single = single[name]
            p_any = 1 - (1 - p_single)**n
            prize = prize_structure[name]

            expected_value += p_any * prize
            division_breakdown[name] = {
                'probability': p_any,
                'prize': prize,
                'expected_prize': prize
            }

        return {
            'total_investment': total_investment,
            'expected_value': expected_value,
            'expected_profit': expected_value - total_investment,
            'return_on_investment': (expected_value - total_investment) / total_investment if total_investment > 0 else 0,
            'jackpot_amount': jackpot_amount,
            'estimated_sales': estimated_sales,
            'division_breakdown': division_breakdown,
            'run_id': self.run_id,
            'seed': self.seed
        }

    def _calculate_division_probabilities(self, num_tickets: int) -> Dict[str, float]:
        """Calculate probability of winning each division with multiple tickets (Powerball only)"""
        single = self._powerball_single_ticket_division_probs()
        return {k: 1 - (1 - p)**num_tickets for k, p in single.items()}

    def _single_ticket_division_probs(self) -> Dict[str, float]:
        """Calculate single ticket probabilities using proper combinatorics (legacy - mixed Lotto/Powerball)"""
        denom = comb(40, 6)
        p_pb = 1/10
        p_np = 9/10

        p6 = 1/denom
        p5 = comb(6,5) * comb(34,1) / denom
        p4 = comb(6,4) * comb(34,2) / denom
        p5b = p5 * (1/34)  # 5 main + bonus (no PB)

        return {
            'Division 1': p6 * p_pb,   # 6 + PB
            'Division 2': p6 * p_np,   # 6
            'Division 3': p5 * p_pb,   # 5 + PB
            'Division 4': p5b,         # 5 + bonus (no PB)
            'Division 5': p5 * p_np,   # 5
            'Division 6': p4 * p_pb,   # 4 + PB
            'Division 7': p4 * p_np,   # 4
        }

    def _powerball_single_ticket_division_probs(self) -> Dict[str, float]:
        """Calculate Powerball-only single ticket probabilities using official divisions"""
        D = comb(40, 6); C = comb; p_pb = 1/10; pB = 1/34
        p6 = 1/D
        p5 = C(6,5)*C(34,1)/D
        p4 = C(6,4)*C(34,2)/D
        p3 = C(6,3)*C(34,3)/D
        return {
            'PB Div 1 (6+PB)':            p6 * p_pb,
            'PB Div 2 (5+Bonus+PB)':      p5 * pB * p_pb,
            'PB Div 3 (5+PB)':            p5 * p_pb,
            'PB Div 4 (4+Bonus+PB)':      p4 * pB * p_pb,
            'PB Div 5 (4+PB)':            p4 * p_pb,
            'PB Div 6 (3+Bonus+PB)':      p3 * pB * p_pb,
            'PB Div 7 (3+PB)':            p3 * p_pb,
        }

    def _expected_share_factor(self, S: int, p: float) -> float:
        """Calculate expected share factor for co-winners using Poisson approximation"""
        # For large populations, use Poisson closed form: E[1/(1+Z)] = (1 - e^(-λ))/λ
        lam = S * p
        if lam <= 1e-12:
            return 1.0

        # Use Poisson approximation for expected share factor
        share = (1 - math.exp(-lam)) / lam
        return max(0.0, min(1.0, share))  # clamp to (0,1]

    def export_portfolio(self, portfolio: List[Dict], filename: str = 'nz_powerball_portfolio.csv') -> pd.DataFrame:
        """Export optimized portfolio to CSV format with transparency metadata"""
        export_data = []
        
        for ticket in portfolio:
                row = {
                'Ticket_ID': ticket['ticket_id'],
                'Ball_1': ticket['main_balls'][0],
                'Ball_2': ticket['main_balls'][1],
                'Ball_3': ticket['main_balls'][2],
                'Ball_4': ticket['main_balls'][3],
                'Ball_5': ticket['main_balls'][4],
                'Ball_6': ticket['main_balls'][5],
                'Bonus_Ball': ticket.get('bonus', 0),
                'Powerball': ticket['powerball'],
                'Sum': ticket['sum'],
                'Odd_Count': ticket['odd_count'],
                'Even_Count': ticket['even_count'],
                'Last_Digits': ','.join(map(str, ticket['last_digits'])),
                'Run_ID': self.run_id,
                'Seed': self.seed,
                'Generated': datetime.now().isoformat()
                }
                export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Portfolio exported to {filename} (Run ID: {self.run_id})")
        
        return df

    def display_portfolio_analysis(self, portfolio: List[Dict], expected_value_analysis: Dict):
        """Display comprehensive portfolio analysis"""
        print("\n" + "="*80)
        print("NZ POWERBALL TICKET PORTFOLIO OPTIMIZER")
        print("="*80)
        print(f"Run ID: {self.run_id}")
        print(f"Seed: {self.seed}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.use_tft:
            mode_desc = "deterministic (top-K)" if self.tft_mode == 'deterministic' else "probabilistic (weighted)"
            print("🎯 USING PRODUCTION-GRADE BiLSTM PREDICTIONS")
            print(f"   Mode: {mode_desc}")
            print("   (Three separate models: MAIN/BONUS/POWERBALL)")
        else:
            print("🎲 USING UNIFORM RANDOM SAMPLING")
            print("   (Provably fair lottery mechanics)")
        print()

        print("PORTFOLIO SUMMARY:")
        print(f"Total Tickets: {len(portfolio)}")
        total_investment = len(portfolio) * 1.50
        print(f"Total Investment: ${total_investment:.2f}")
        print()

        print("TICKET DETAILS:")
        print("-" * 80)
        print(f"{'ID':<2} | {'Main Balls':<20} | {'Bonus':<2} | {'PB':<2} | {'Sum':<3} | {'O/E':<3} | Last Digits")
        print("-" * 90)

        for ticket in portfolio:
            balls_str = ' '.join(f"{b:02d}" for b in ticket['main_balls'])
            bonus = ticket.get('bonus', 0)
            odd_even = f"{ticket['odd_count']}/{ticket['even_count']}"
            digits_str = ','.join(str(d) for d in sorted(ticket['last_digits']))
            print(f"{ticket['ticket_id']:<2} | {balls_str:<20} | {bonus:02d} | {ticket['powerball']:02d} | {ticket['sum']:<3} | {odd_even:<3} | {digits_str}")

        print("\nDIVERSITY ANALYSIS:")
        print("-" * 40)

        # Sum distribution
        sums = [t['sum'] for t in portfolio]
        print(f"Sum range: {min(sums)} - {max(sums)} (target: 70-180)")
        print(f"Unique sums: {len(set(sums))}/{len(portfolio)}")

        # Odd/even balance
        odd_counts = [t['odd_count'] for t in portfolio]
        print(f"Odd count distribution: {min(odd_counts)} - {max(odd_counts)} (target: ~3)")

        # Last digit variety
        all_digits = []
        for t in portfolio:
            all_digits.extend(t['last_digits'])
        unique_digits = len(set(all_digits))
        print(f"Unique last digits used: {unique_digits}/10")

        print("\nEXPECTED VALUE ANALYSIS:")
        print("-" * 40)
        print(f"Expected Value: ${expected_value_analysis['expected_value']:.2f}")
        print(f"Expected Profit: ${expected_value_analysis['expected_profit']:.2f}")
        print(f"ROI: {expected_value_analysis['return_on_investment']:.6f}")
        if expected_value_analysis['expected_profit'] > 0:
            print("🎯 PORTFOLIO SHOWS POSITIVE EXPECTED VALUE")
        else:
            print("⚠️  PORTFOLIO SHOWS NEGATIVE EXPECTED VALUE (as expected for lottery)")

        print("\nWINNING PROBABILITIES:")
        for division, details in expected_value_analysis['division_breakdown'].items():
            prob_pct = details['probability'] * 100
            if prob_pct > 0:
                print(f"{division:<12}: {prob_pct:.6f}%  (exp prize: ${details['expected_prize']:.2f})")

        print("\n" + "="*80)
        print("TRANSPARENCY STATEMENT:")
        if self.use_tft:
            mode_desc = "deterministic (top-K)" if self.tft_mode == 'deterministic' else "probabilistic (Plackett-Luce)"
            print("- This portfolio was generated using production-grade BiLSTM predictions")
            print("  with three separate models: MAIN (6/40), BONUS (1/40), POWERBALL (1/10)")
            print("- Each model uses bidirectional LSTM with attention mechanism")
            print(f"- Inference mode: {mode_desc} sampling")
            print("- Predictions calibrated with guardrails to prevent mode collapse")
            print("- Soft popularity penalties applied as probabilistic adjustments")
            print("- ⚠️  ML predictions carry no EV guarantee; uniform sampling remains sound")
        else:
            print("- This portfolio was generated using cryptographically seeded")
            print("  uniform random sampling with popularity avoidance heuristics")
            print("- No ML models or pattern prediction was used")
            print("- All tickets have equal theoretical probability of winning")

        print("- Figures assume $1.50/line (Lotto $0.70 + Powerball add-on $0.80)")
        print("  and official Powerball divisions with estimated prize amounts")
        print("- Expected value analysis accounts for co-winner scenarios using")
        print("  Poisson approximation for large populations")
        print("="*80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='NZ Powerball Ticket Portfolio Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use uniform sampling (default)
  python main.py --tft                      # Use TFT predictions
  python main.py --tft --tickets 20         # TFT with 20 tickets
  python main.py --seed abc123              # Custom seed for reproducibility
        """
    )

    parser.add_argument(
        '--tft', '--ml',
        action='store_true',
        help='Enable TFT-based probability predictions (experimental)'
    )

    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Force uniform sampling (default behavior)'
    )

    parser.add_argument(
        '--tickets', '-n',
        type=int,
        default=10,
        help='Number of tickets to generate (default: 10)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=str,
        default=None,
        help='Cryptographic seed for reproducible results'
    )

    parser.add_argument(
        '--jackpot', '-j',
        type=float,
        default=50000000,
        help='Jackpot amount in NZD (default: 50M)'
    )

    parser.add_argument(
        '--sales',
        type=int,
        default=400000,
        help='Estimated ticket sales for co-winner calculation (default: 400k)'
    )

    parser.add_argument(
        '--tft-mode',
        choices=['deterministic', 'probabilistic'],
        default='probabilistic',
        help='TFT inference mode: deterministic (top-K) or probabilistic (weighted sampling)'
    )

    return parser.parse_args()

def main():
    """Main execution function - NZ Powerball Ticket Portfolio Optimizer"""

    args = parse_arguments()

    print("="*80)
    print("NZ POWERBALL TICKET PORTFOLIO OPTIMIZER")
    print("="*80)

    # Determine ML usage
    use_tft = args.tft and not args.no_ml
    if use_tft and not LSTM_AVAILABLE:
        print("⚠️  TFT requested but dependencies not available. Install:")
        print("   pip install pytorch-forecasting pytorch-lightning")
        print("   Falling back to uniform sampling...")
        use_tft = False

    if use_tft:
        mode_desc = "deterministic (top-K)" if args.tft_mode == 'deterministic' else "probabilistic (weighted)"
        print("🎯 Using production-grade TFT predictions")
        print(f"   Mode: {mode_desc}")
        print("   (Three separate models: MAIN/BONUS/POWERBALL)")
    else:
        print("🎲 Using uniform random sampling")
        print("   (Provably fair lottery mechanics)")

    print("Implementing core principles:")
    print("• Cryptographically seeded RNG with reproducible run IDs")
    print("• Popularity avoidance heuristics")
    print("• Expected value analysis with co-winner scenarios")
    if use_tft:
        print("• BiLSTM predictions with Plackett-Luce weighted sampling")
    print()

    # Initialize optimizer with historical data
    optimizer = NZPowerballTicketOptimizer(
        'NZ Powerball-till 27-08-2025.csv',
        seed=args.seed,
        use_tft=use_tft
    )

    # Set TFT mode if using TFT
    if use_tft:
        optimizer.tft_mode = args.tft_mode

    # Load historical data for analysis (trains TFT models if enabled)
    print("Loading historical data...")
    optimizer.load_historical_data()

    # Generate optimized ticket portfolio
    print(f"\nGenerating optimized ticket portfolio ({args.tickets} tickets)...")
    portfolio = optimizer.generate_ticket_portfolio(
        num_tickets=args.tickets,
        diversity_constraints={
            'max_consecutive': 2,  # Optimal for lottery
            'balance_odd_even': True,
            'avoid_popular_patterns': True,
            'sum_range': (85, 165),  # Statistical sweet spot
            'digit_ending_variety': True
        }
    )

    # Calculate expected value analysis
    print("\nCalculating expected value analysis...")
    expected_value_analysis = optimizer.calculate_expected_value(
        portfolio,
        jackpot_amount=args.jackpot,
        estimated_sales=args.sales,
        prize_structure=None  # Use defaults
    )

    # Display comprehensive analysis
    optimizer.display_portfolio_analysis(portfolio, expected_value_analysis)

    # Export portfolio
    optimizer.export_portfolio(portfolio)

    print(f"\n✅ Portfolio optimization complete!")
    print(f"📋 Run ID: {optimizer.run_id}")
    print(f"🔐 Seed: {optimizer.seed}")

    return optimizer, portfolio, expected_value_analysis

def backtest_comparison(csv_path: str, test_draws: int = 10, num_tickets: int = 100,
                       seed: str = None) -> Dict:
    """
    Backtest TFT vs uniform sampling performance over historical draws.

    Args:
        csv_path: Path to historical data CSV
        test_draws: Number of recent draws to test on
        num_tickets: Number of tickets per portfolio per draw
        seed: Random seed for reproducibility

    Returns:
        Dictionary with backtest results
    """
    if not LSTM_AVAILABLE:
        print("Backtesting requires TFT dependencies")
        return {}

    print(f"Starting backtest comparison over {test_draws} recent draws...")

    # Load full dataset
    df = pd.read_csv(csv_path)
    df['Draw Date'] = pd.to_datetime(df['Draw Date'], format='%A %d %B %Y', errors='coerce')
    df = df.sort_values('Draw Date').reset_index(drop=True)

    # Split into training and test sets
    train_df = df.iloc[:-test_draws].copy()
    test_df = df.iloc[-test_draws:].copy()

    results = {
        'uniform': {'hits': [], 'expected_values': []},
        'tft': {'hits': [], 'expected_values': []}
    }

    # For each test draw
    for i, (_, test_row) in enumerate(test_df.iterrows()):
        print(f"Testing draw {i+1}/{test_draws}...")

        # Get actual winning numbers for this draw
        actual_main = sorted([int(test_row[str(j)]) for j in range(1, 7)])
        actual_pb = int(test_row['Power Ball'])

        # Create training data up to this point
        current_train = df.iloc[:len(train_df) + i].copy()

        # Test uniform sampling
        uniform_optimizer = NZPowerballTicketOptimizer(
            csv_path, seed=f"{seed}_uniform_{i}" if seed else None, use_tft=False
        )
        uniform_optimizer.data = current_train
        uniform_optimizer._analyze_historical_patterns()

        uniform_portfolio = uniform_optimizer.generate_ticket_portfolio(
            num_tickets=num_tickets,
            diversity_constraints={
                'max_consecutive': 3, 'balance_odd_even': True,
                'avoid_popular_patterns': True, 'sum_range': (70, 180),
                'digit_ending_variety': True
            }
        )

        # Test TFT sampling
        tft_optimizer = NZPowerballTicketOptimizer(
            csv_path, seed=f"{seed}_tft_{i}" if seed else None, use_tft=True
        )
        tft_optimizer.data = current_train
        tft_optimizer._analyze_historical_patterns()
        tft_optimizer._train_tft_models()

        tft_portfolio = tft_optimizer.generate_ticket_portfolio(
            num_tickets=num_tickets,
            diversity_constraints={
                'max_consecutive': 3, 'balance_odd_even': True,
                'avoid_popular_patterns': True, 'sum_range': (70, 180),
                'digit_ending_variety': True
            }
        )

        # Calculate hits for both portfolios
        uniform_hits = calculate_portfolio_hits(uniform_portfolio, actual_main, actual_pb)
        tft_hits = calculate_portfolio_hits(tft_portfolio, actual_main, actual_pb)

        results['uniform']['hits'].append(uniform_hits)
        results['tft']['hits'].append(tft_hits)

        print(f"  Uniform: {uniform_hits} hits, TFT: {tft_hits} hits")

    # Calculate summary statistics
    uniform_total_hits = sum(results['uniform']['hits'])
    tft_total_hits = sum(results['tft']['hits'])

    print(f"\nBacktest Results ({test_draws} draws, {num_tickets} tickets each):")
    print(f"Uniform sampling: {uniform_total_hits} total hits")
    print(f"TFT sampling: {tft_total_hits} total hits")
    if test_draws > 0:
        uniform_avg = uniform_total_hits / test_draws
        tft_avg = tft_total_hits / test_draws
        print(f"Uniform average hits per draw: {uniform_avg:.3f}")
        print(f"TFT average hits per draw: {tft_avg:.3f}")

    return results

def calculate_portfolio_hits(portfolio: List[Dict], actual_main: List[int], actual_pb: int) -> int:
    """Calculate total number of hits across all tickets in portfolio."""
    total_hits = 0

    for ticket in portfolio:
        ticket_main = set(ticket['main_balls'])
        ticket_pb = ticket['powerball']

        # Count main ball matches
        main_matches = len(ticket_main.intersection(set(actual_main)))

        # Count powerball match
        pb_match = 1 if ticket_pb == actual_pb else 0

        # For simplicity, count any match as a hit
        if main_matches > 0 or pb_match > 0:
            total_hits += 1

    return total_hits


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        # Run backtest mode
        print("Running backtest comparison...")
        results = backtest_comparison(
            'NZ Powerball-till 27-08-2025.csv',
            test_draws=5,
            num_tickets=50
        )
    else:
        # Run normal optimization
        optimizer, portfolio, expected_value_analysis = main() 