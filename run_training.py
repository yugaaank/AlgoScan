from algorithm import CryptoIdentifier

if __name__ == '__main__':
    print("Starting model training...")
    identifier = CryptoIdentifier()
    identifier.train_model('cryptography_dataset_generated.csv')
    print("Model training finished.")
