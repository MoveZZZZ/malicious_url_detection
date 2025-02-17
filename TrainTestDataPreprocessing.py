from LogSystem import LogFileCreator
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import tensorflow as tf
from scipy.sparse import hstack
class TrainTestDataPreprocessing:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)


    def create_X_and_Y(self, data):
        X = data.drop(['type'], axis = 1)
        y = data['type']
        return X,y

    def check_class_balance(self, y):
        self.LogCreator.print_and_write_log("Checking class distribution...")
        self.LogCreator.print_and_write_log("Class distribution:", Counter(y))

    def tokenize_urls(self, urls, tokenizer, max_length=128):
        self.LogCreator.print_and_write_log("Tokenizing URLs...")
        if isinstance(urls, (np.ndarray, pd.Series)):
            urls = urls.tolist()
        elif not isinstance(urls, list):
            urls = [str(url) for url in urls]
        print(f"Type of input: {type(urls)}, Example: {urls[:3]}")
        tokenized_data = {"input_ids": [], "attention_mask": []}

        for url in tqdm(urls, desc="Tokenizing Progress"):
            tokenized = tokenizer(url, padding="max_length", truncation=True, max_length=max_length,
                                  return_tensors="tf")
            tokenized_data["input_ids"].append(tokenized["input_ids"][0].numpy())
            tokenized_data["attention_mask"].append(tokenized["attention_mask"][0].numpy())

        tokenized_data["input_ids"] = tf.convert_to_tensor(tokenized_data["input_ids"])
        tokenized_data["attention_mask"] = tf.convert_to_tensor(tokenized_data["attention_mask"])
        self.LogCreator.print_and_write_log(f"Tokenized shape: input_ids={tokenized_data['input_ids'].shape}, attention_mask={tokenized_data['attention_mask'].shape}")
        return tokenized_data

    def extract_tfidf_features(self,urls):
        self.LogCreator.print_and_write_log("Extracting TF-IDF features...")
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
        features = vectorizer.fit_transform(urls)
        self.LogCreator.print_and_write_log(f"TF-IDF shape: {features.shape}")
        return features

    def prepare_data_bert_2(self, X, y, tokenizer):
        self.LogCreator.print_and_write_log("Preparing data...")
        X_tokenized = self.tokenize_urls(X["url"], tokenizer)
        #X_features = self.extract_tfidf_features(X["url"])
        X_input_ids = X_tokenized["input_ids"].numpy()
        X_input_att = X_tokenized["attention_mask"].numpy()
        #X_combined = hstack([X_input_ids, X_features])
        #print(f"Final feature shape: {X_combined.shape}")
        X_train, X_test, y_train, y_test = self.split_data_for_train_and_validation_bert(X_input_ids, X_input_att, y, 0.2, 42)
        for key, value in X_train.items():
            print(f"{key} shape: {value.shape}")
        return X_train, X_test, y_train, y_test

    def split_data_for_train_and_validation_bert(self,  X_input_ids, X_attention_mask, y, _test_size, _random_state):
        self.LogCreator.print_and_write_log("Start split data on train and test bert")
        spit_time_start = time.time()
        X_input_ids_np = np.array(X_input_ids)
        X_attention_mask_np = np.array(X_attention_mask)
        X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
            X_input_ids_np, X_attention_mask_np, y, test_size=_test_size, random_state=_random_state
        )
        X_train = {
            "input_ids": X_train_ids,
            "attention_mask": X_train_mask
        }
        X_test = {
            "input_ids": X_test_ids,
            "attention_mask": X_test_mask
        }
        spit_time_end = time.time()
        self.LogCreator.print_and_write_log(f"End split data on train and test bert. "
                                            f"Time to split: {self.LogCreator.count_time(spit_time_start, spit_time_end):.2f} s.\n"
                                            f"{self.LogCreator.string_spit_stars}")
        return X_train, X_test, y_train, y_test

    def split_data_for_train_and_validation(self, X, y, _test_size, _random_state):
        self.LogCreator.print_and_write_log("Start split data on train and test")
        spit_time_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_size, random_state=_random_state)
        spit_time_end = time.time()
        self.LogCreator.print_and_write_log(f"End split data on train and test. "
                                            f"Time to split: {self.LogCreator.count_time(spit_time_start, spit_time_end):.2f} s.\n"
                                            f"{self.LogCreator.string_spit_stars}")
        return X_train, X_test, y_train, y_test

    def create_scaler(self, X):
        self.LogCreator.print_and_write_log("Start creating scaler")
        time_to_create_scaler_start = time.time()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)

        time_to_create_scaler_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End creating scaler. Time to create scaler: "
            f"{self.LogCreator.count_time(time_to_create_scaler_start, time_to_create_scaler_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}"
        )
        return scaler
    def scale_data(self, scaler,X_train, X_test=None):
        self.LogCreator.print_and_write_log("Start scaling data")
        time_to_scale_start = time.time()

        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
        if X_test is not None:
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        else:
            X_test_scaled = None
        time_to_scale_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End scaling data. Time to scale data: "
            f"{self.LogCreator.count_time(time_to_scale_start, time_to_scale_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}"
        )
        return X_train_scaled, X_test_scaled