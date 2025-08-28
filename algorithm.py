# ============================================
# Cryptographic Algorithm Identifier (Clean Rewrite)
# - Single, consistent feature pipeline (NumPy-first)
# - No duplicate function definitions
# - Robust against NaNs and type issues
# - Correct CatBoost Pool usage
# ============================================

# ====== 1. IMPORTS ======
import os
import math
from collections import Counter

import numpy as np
import pandas as pd

from scipy.stats import entropy as scipy_entropy, skew as scipy_skew, kurtosis as scipy_kurtosis

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier, Pool
import joblib

# ============================================
# 2. HELPERS & FEATURE EXTRACTION (NumPy-first)
# ============================================

# ---- Helper: Convert Hex to Bytes ----
def hex_to_bytes(hex_string: str) -> bytes:
    """
    Converts a hexadecimal string into a bytes object.
    Returns an empty bytes object if the hex string is invalid.
    """
    try:
        return bytes.fromhex(hex_string)
    except Exception:
        return b""

# ---- Helper: Clean Hex ----
def clean_hex_string(input_string: str) -> str:
    """Remove all non-hexadecimal characters and return cleaned string."""
    s = ''.join(c for c in str(input_string).strip() if c in '0123456789abcdefABCDEF')
    # Ensure even length for fromhex
    return s if len(s) % 2 == 0 else ('0' + s)

# ---- Byte frequency over uint8 array ----
def calculate_byte_frequency(byte_arr: np.ndarray) -> np.ndarray:
    """
    Expects 1D np.ndarray dtype=uint8. Returns normalized frequencies for values 0..255.
    """
    if byte_arr is None or byte_arr.size == 0:
        return np.zeros(256, dtype=np.float64)
    counts = np.bincount(byte_arr, minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.zeros(256, dtype=np.float64)
    return counts / total

# ---- Shannon Entropy (0..1) ----
def calculate_entropy(data: np.ndarray | bytes) -> float:
    """
    Compute Shannon entropy normalized to [0,1] (divide by 8 since base-2 entropy of byte is <= 8 bits).
    Accepts np.ndarray[uint8] or bytes.
    """
    if data is None:
        return 0.0
    if isinstance(data, (bytes, bytearray)):
        byte_arr = np.frombuffer(data, dtype=np.uint8)
    else:
        byte_arr = np.asarray(data, dtype=np.uint8).reshape(-1)

    if byte_arr.size == 0:
        return 0.0

    freqs = calculate_byte_frequency(byte_arr)
    non_zero = freqs[freqs > 0]
    if non_zero.size == 0:
        return 0.0
    return float(scipy_entropy(non_zero, base=2) / 8.0)

# ---- Base64 Charset Check ----
_BASE64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")

def is_base64_charset(data: np.ndarray | bytes) -> float:
    """
    Returns 1.0 if all bytes are valid Base64 chars, else 0.0.
    """
    try:
        if isinstance(data, np.ndarray):
            b = data.tobytes()
        else:
            b = bytes(data)
        s = b.decode('ascii')
        return 1.0 if all(c in _BASE64_CHARS for c in s) else 0.0
    except Exception:
        return 0.0

# ---- Block Pattern Analysis ----
def analyze_block_patterns(byte_arr: np.ndarray, block_size: int = 16) -> dict:
    """
    Analyze adjacent-block similarity, lag-2 inter-block correlation, boundary entropy, and intra-block std.
    Expects 1D uint8 np.ndarray.
    """
    n = byte_arr.size
    if n < 2 * block_size:
        return {
            'block_similarity': 0.0,
            'inter_block_correlation': 0.0,
            'block_boundary_entropy': 0.0,
            'intra_block_std': 0.0,
        }

    # Create list of blocks (views)
    blocks = [byte_arr[i:i+block_size] for i in range(0, n, block_size) if i + block_size <= n]

    # Adjacent block similarity (Pearson corr)
    sims = []
    for i in range(len(blocks) - 1):
        b1 = blocks[i].astype(np.float64)
        b2 = blocks[i+1].astype(np.float64)
        if np.std(b1) > 0 and np.std(b2) > 0:
            val = np.corrcoef(b1, b2)[0, 1]
            if not np.isfinite(val):
                val = 0.0
        else:
            val = 0.0
        sims.append(float(val))
    block_similarity = float(np.mean(sims)) if sims else 0.0

    # Inter-block correlation with lag 2
    lag2 = []
    for i in range(len(blocks) - 2):
        b1 = blocks[i].astype(np.float64)
        b2 = blocks[i+2].astype(np.float64)
        if np.std(b1) > 0 and np.std(b2) > 0:
            val = np.corrcoef(b1, b2)[0, 1]
            if not np.isfinite(val):
                val = 0.0
        else:
            val = 0.0
        lag2.append(float(val))
    inter_block_corr = float(np.mean(lag2)) if lag2 else 0.0

    # Boundary entropy: every 16th byte (index 15, 31, ...)
    boundaries = byte_arr[block_size-1::block_size]
    block_boundary_entropy = calculate_entropy(boundaries)

    # Intra-block std: mean of per-block std
    intra_stds = [float(np.std(b.astype(np.float64))) for b in blocks]
    intra_block_std = float(np.mean(intra_stds)) if intra_stds else 0.0

    return {
        'block_similarity': block_similarity,
        'inter_block_correlation': inter_block_corr,
        'block_boundary_entropy': float(block_boundary_entropy),
        'intra_block_std': intra_block_std,
    }

# ---- Temporal Pattern Analysis ----
def analyze_temporal_patterns(byte_arr: np.ndarray) -> dict:
    """Simple first-difference stats over the byte stream."""
    if byte_arr is None or byte_arr.size < 32:
        return {
            'time_domain_mean': 0.0,
            'time_domain_var': 0.0,
            'time_domain_skew': 0.0,
            'time_domain_kurt': 0.0,
        }
    x = byte_arr.astype(np.float64)
    diffs = np.diff(x)
    ad = np.abs(diffs)
    return {
        'time_domain_mean': float(np.mean(ad)),
        'time_domain_var': float(np.var(ad)),
        'time_domain_skew': float(scipy_skew(ad)),
        'time_domain_kurt': float(scipy_kurtosis(ad)),
    }

# ---- Main Feature Extraction ----
def extract_features(hex_data_string: str) -> dict:
    """Extract a set of numeric, stable features from a hex string."""
    cleaned = clean_hex_string(hex_data_string)
    b = hex_to_bytes(cleaned)
    arr = np.frombuffer(b, dtype=np.uint8)

    # 1) Statistical
    features: dict[str, float] = {}
    features['shannon_entropy'] = float(calculate_entropy(arr))
    if arr.size > 0:
        freqs = calculate_byte_frequency(arr)
        features['freq_std'] = float(np.std(freqs))
        features['is_uniform'] = 1.0 if float(np.std(freqs)) < 0.005 else 0.0
        # Bit density
        bits = np.unpackbits(arr)
        features['norm_one_bit_count'] = float(np.mean(bits))
    else:
        features['freq_std'] = 0.0
        features['is_uniform'] = 0.0
        features['norm_one_bit_count'] = 0.0

    features['len_bucket_64'] = float(min(arr.size // 64, 8))
     # 2) Structural
    features['len_bucket_64'] = float(min(arr.size // 64, 8))

    # --- Additional structure-aware features ---
    def _hamming(a, b):
        return np.unpackbits(np.bitwise_xor(a, b)).sum()

    def _interblock_hamming_stats(arr, block=16):
        if arr.size < 2*block:
            return 0.0, 0.0
        blocks = arr[:(arr.size//block)*block].reshape(-1, block)
        dists = np.array([_hamming(blocks[i], blocks[i+1]) for i in range(len(blocks)-1)], dtype=np.float64)
        return float(dists.mean()), float(dists.var())

    mean_hd, var_hd = _interblock_hamming_stats(arr)
    features['interblock_hamming_mean'] = mean_hd
    features['interblock_hamming_var']  = var_hd

    def _ecb_repetition_score(arr, block=16):
        if arr.size < 2*block:
            return 0.0
        blocks = arr[:(arr.size//block)*block].reshape(-1, block)
        views = [bytes(b) for b in blocks]
        unique = len(set(views))
        return 1.0 - (unique / len(views))  # 0=no repeats, 1=all same

    features['ecb_repetition_score'] = _ecb_repetition_score(arr)

    def _bit_autocorr(arr, max_lag=8):
        if arr.size < 64:
            return [0.0]*max_lag
        bits = np.unpackbits(arr).astype(np.float64)
        bits = bits - bits.mean()
        var = bits.var()
        if var == 0:
            return [0.0]*max_lag
        ac = []
        for lag in range(1, max_lag+1):
            x = bits[:-lag]
            y = bits[lag:]
            ac.append(float((x*y).mean() / var))
        return ac

    ac = _bit_autocorr(arr, max_lag=8)
    for i, v in enumerate(ac, 1):
        features[f'bit_autocorr_lag{i}'] = v
    if arr.size > 1:
        pairs = arr[:-1].astype(np.uint16) << 8 | arr[1:]
        counts = np.bincount(pairs, minlength=65536).astype(np.float64)
        probs_ng = counts[counts > 0] / counts.sum()
        features['bigram_entropy'] = float(scipy_entropy(probs_ng, base=2) / 16.0)  # normalized
    else:
        features['bigram_entropy'] = 0.0
    # 3) Block Patterns
    features.update(analyze_block_patterns(arr))

    # 4) Temporal Patterns
    features.update(analyze_temporal_patterns(arr))

    return features

# ============================================
# 3. MODEL TRAINING & IDENTIFICATION CLASS
# ============================================
def _apply_post_hoc_guards(hex_input_data, labels, probs):
    b = bytes.fromhex(clean_hex_string(hex_input_data))
    n = len(b)

    # RSA guard: common modulus sizes in bytes
    rsa_lengths = {128, 256, 384, 512}
    for i, lbl in enumerate(labels):
        if lbl.upper() == 'RSA' and n not in rsa_lengths:
            probs[i] = 1e-9

    # SHA256 guard: exactly 32 bytes
    for i, lbl in enumerate(labels):
        if lbl.upper() == 'SHA256' and n != 32:
            probs[i] = 1e-9

    # Renormalize
    s = probs.sum()
    if s == 0:
        probs[:] = 1.0 / len(probs)
    else:
        probs[:] = probs / s
    return probs


class CryptoIdentifier:
    def __init__(self):
        self.model: CatBoostClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] | None = None
        self.label_encoder: LabelEncoder | None = None
        self.load_model()

    # ---- Load persisted artifacts ----
    def load_model(self):
        try:
            self.model = joblib.load('crypto_model.pkl')
            self.scaler = joblib.load('crypto_scaler.pkl')
            self.feature_names = joblib.load('crypto_features.pkl')
            self.label_encoder = joblib.load('crypto_label_encoder.pkl')
            print('Pre-trained model loaded successfully.')
        except Exception:
            print('No pre-trained model found. A new one will be trained.')
            self.model = None
            self.scaler = None
            self.feature_names = None
            self.label_encoder = None

    # ---- Evaluate on test set ----
    def test_model(self, dataset_path: str):
        if any(v is None for v in [self.model, self.scaler, self.feature_names, self.label_encoder]):
            print("Model artifacts not loaded. Cannot run evaluation.")
            return

        print(f"\nEvaluating pre-trained model using dataset from '{dataset_path}'...")
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print('Error: The CSV dataset file was not found. Cannot perform evaluation.')
            return

        df.rename(columns={'Ciphertext': 'hex_data', 'Algorithm': 'algorithm'}, inplace=True)
        df['algorithm'] = df['algorithm'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)
        df['hex_data'] = df['hex_data'].astype(str).map(clean_hex_string)
        df.dropna(subset=['hex_data', 'algorithm'], inplace=True)
        df = df[df['hex_data'].str.len() > 0].copy()

        # Feature extraction
        features_list = []
        for idx, row in df.iterrows():
            feats = extract_features(row['hex_data'])
            features_list.append(feats)

        X = pd.DataFrame(features_list)
        # Reorder columns to match the training feature set
        X = X.reindex(columns=self.feature_names, fill_value=0.0)
        X.replace([np.inf, -np.inf], 0.0, inplace=True)
        X.fillna(0.0, inplace=True)

        y = df['algorithm'].values
        # Only keep labels that were present during training
        valid_labels = set(self.label_encoder.classes_)
        df_filtered = df[df['algorithm'].isin(valid_labels)]
        X = X.loc[df_filtered.index]
        y = df_filtered['algorithm'].values

        y_enc = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)

        y_pred = self.model.predict(X_scaled)
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)

        print('\n--- Model Evaluation on Provided Test Data ---')
        acc = accuracy_score(y_enc, y_pred)
        print(f'Accuracy: {acc * 100:.2f}%')

        print('\nClassification Report:')
        target_names = list(self.label_encoder.classes_)
        print(classification_report(y_enc, y_pred, target_names=target_names, zero_division=0))

    # ---- Train ----
    def train_model(self, dataset_path: str):
        print(f"Loading dataset from '{dataset_path}'...")
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print('Error: The CSV dataset file was not found. Please check the path and filename.')
            return

        # Standardize column names
        df.rename(columns={'Ciphertext': 'hex_data', 'Algorithm': 'algorithm'}, inplace=True)

        # Clean text columns
        df['algorithm'] = df['algorithm'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)
        df['hex_data'] = df['hex_data'].astype(str).map(clean_hex_string)

        # Drop empties
        df = df[df['hex_data'].str.len() > 0].copy()
        df.dropna(subset=['hex_data', 'algorithm'], inplace=True)

        print('Successfully loaded and prepared the dataset.')

        # Feature extraction
        print('Extracting features from dataset...')
        features_list = []
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  - Processing sample {idx}/{len(df)}")
            feats = extract_features(row['hex_data'])
            features_list.append(feats)

        X = pd.DataFrame(features_list)
        # sanitize
        X.replace([np.inf, -np.inf], 0.0, inplace=True)
        X.fillna(0.0, inplace=True)
        y = df['algorithm'].values

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = list(X.columns)

        print('Splitting data into training and testing sets...')
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = {int(k): float(v) for k, v in zip(np.unique(y_train), class_weights)}

        print('Training CatBoostClassifier model...')
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            od_type='Iter',
            early_stopping_rounds=50,
            random_seed=42,
            verbose=100,
            thread_count=-1,
            class_weights=weight_dict,
        )

        eval_pool = Pool(X_test, y_test)
        self.model.fit(X_train, y_train, eval_set=eval_pool, early_stopping_rounds=50, verbose=100)

        print('Model training complete.')

        # Evaluation
        y_pred = self.model.predict(X_test)
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)

        print('\n--- Model Evaluation on Test Data ---')
        acc = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {acc * 100:.2f}%')

        print('\nClassification Report:')
        target_names = list(self.label_encoder.classes_)
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Persist artifacts
        joblib.dump(self.model, 'crypto_model.pkl')
        joblib.dump(self.scaler, 'crypto_scaler.pkl')
        joblib.dump(self.feature_names, 'crypto_features.pkl')
        joblib.dump(self.label_encoder, 'crypto_label_encoder.pkl')
        print('Model, scaler, and encoder saved to disk.')

    # ---- Inference ----
    def identify_algorithm(self, hex_input_data: str):
        if any(v is None for v in [self.model, self.scaler, self.feature_names, self.label_encoder]):
            print('No trained model found. Please train a new model first.')
            return []
        # Stage 1: RSA/SHA256 trivial check
        b = bytes.fromhex(clean_hex_string(hex_input_data))
        n = len(b)

        if n in {128, 256, 384, 512}:
            print("\nDetected RSA by length.")
            return [("RSA", 1.0)]
        elif n == 32:
            print("\nDetected SHA256 by length.")
            return [("SHA256", 1.0)]

        print(f"\nAnalyzing input data (length: {len(hex_input_data)} hex chars)...")
        feats = extract_features(hex_input_data)
        df_in = pd.DataFrame([feats])
        df_in = df_in.reindex(columns=self.feature_names, fill_value=0.0)
        x_in = self.scaler.transform(df_in)

        probs = self.model.predict_proba(x_in)[0]
        class_idx = np.arange(len(probs))
        labels = self.label_encoder.inverse_transform(class_idx)

        # Apply RSA/SHA256 sanity checks
        probs = _apply_post_hoc_guards(hex_input_data, labels, probs)

        pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

        print('\n--- Cryptographic Algorithm Identification Results ---')
        for i, (algo, p) in enumerate(pairs, start=1):
            level = 'HIGH' if p > 0.7 else ('MEDIUM' if p > 0.4 else 'LOW')
            print(f'{i}. {algo}: {p*100:.2f}% confidence [{level}]')
        return pairs

# ============================================
# 4. CLI ENTRY POINT
# ============================================
if __name__ == '__main__':
    print('=== Cryptographic Algorithm Identifier ===')
    identifier = CryptoIdentifier()

    dataset_filename = 'cryptography_dataset_generated.csv'
    if identifier.model is None:
        identifier.train_model(dataset_filename)
    else:
        # Prompt for evaluation dataset path if a model is loaded
        eval_path = input(f"Pre-trained model found. Enter the path to the evaluation dataset (e.g., '{dataset_filename}') or press Enter to skip evaluation: ")
        if eval_path:
            identifier.test_model(eval_path)

    if identifier.model is not None:
        try:
            user_input = input('\nEnter data to analyze (hex or Base64-like string of hex chars): ')
        except EOFError:
            user_input = ''

        cleaned = clean_hex_string(user_input)
        if len(cleaned) == 0:
            print('No valid hexadecimal input provided. Exiting.')
        else:
            # Validate hex
            try:
                bytes.fromhex(cleaned)
            except ValueError:
                print('Invalid hexadecimal data. Please enter valid hex characters.')
                raise SystemExit(1)

            results = identifier.identify_algorithm(cleaned)
            if results:
                print(f"\nAnalysis complete. Most likely algorithm: {results[0][0]}")
