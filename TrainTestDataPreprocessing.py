from LogSystem import LogFileCreator
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from tqdm import tqdm
import tensorflow as tf
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.cluster import MiniBatchKMeans

class TrainTestDataPreprocessing:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)

    def create_X_and_Y(self, data):
        X = data.drop(['type'], axis = 1)
        y = data['type']
        return X,y

    def compute_class_weights(self, y_train):
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = np.sum(class_counts)
        class_weights = {cls: total_samples / (len(unique_classes) * count) for cls, count in zip(unique_classes, class_counts)}
        weights_array = np.array([class_weights[i] if i in class_weights else 1.0 for i in range(len(class_weights))], dtype=np.float32)
        #print(f"Class distribution: 0={class_counts[0]}, 1={class_counts[1]}, 2={class_counts[2]}, 3={class_counts[3]}")
        return tf.constant(weights_array, dtype=tf.float32)

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

    def prepare_data_bert_2(self, X, y, tokenizer):
        self.LogCreator.print_and_write_log("Preparing data...")
        X_tokenized = self.tokenize_urls(X["url"], tokenizer)
        X_input_ids = X_tokenized["input_ids"].numpy()
        X_input_att = X_tokenized["attention_mask"].numpy()
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

    def option_preprocessing(self, option, X, y):
        if option in [2, 92]:
            X_resampled, y_resampled = self.smote_oversampling(X,y)
        elif option in [3, 93]:
            X_resampled, y_resampled = self.adasyn_oversampling(X,y)
        elif option in [4, 94]:
            X_resampled, y_resampled = self.smote_tomek(X,y)
        elif option in [5, 95]:
            X_resampled, y_resampled = self.smote_enn(X,y)
        elif option in [6,96]:
            X_resampled, y_resampled = self.random_under(X, y)
        else:
            X_resampled = X
            y_resampled = y
        return X_resampled, y_resampled

    def get_kmeans_centers(self, X, num_centers,batch_size=1000, max_iter=100):
        self.LogCreator.print_and_write_log(f"Start get kmeans centres\n")
        kmeans_centres_get_start = time.time()
        kmeans = MiniBatchKMeans(n_clusters=num_centers, batch_size=batch_size, max_iter=max_iter, random_state=42)
        for _ in tqdm(range(max_iter), desc="MiniBatchKMeans iterations"):
            kmeans = kmeans.partial_fit(X)
        kmeans_centres_get_end = time.time()
        self.LogCreator.print_and_write_log(f"End get kmeans centres\n"
                                            f"Centres: {kmeans.cluster_centers_.shape}\n"
                                            f"Time to get centres: {self.LogCreator.count_time(kmeans_centres_get_start, kmeans_centres_get_end):.2f} s.")
        return kmeans.cluster_centers_

    def smote_oversampling(self, X, y, sampling_strategy='auto', random_state=42):
        self.LogCreator.print_and_write_log(f"Start SMOTE oversampling\n"
                                            f"Data before SMOTE_oversampling: {pd.Series(y).value_counts()}")
        smote_start = time.time()
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        smote_end = time.time()
        self.LogCreator.print_and_write_log(f"End SMOTE oversampling\n"
                                            f"Data after SMOTE_oversampling: {pd.Series(y_resampled).value_counts()}\n"
                                            f"Time to SMOTE_oversampling: {self.LogCreator.count_time(smote_start, smote_end):.2f} s.\n"
                                            f"{self.LogCreator.string_spit_stars}")
        return X_resampled, y_resampled


    def adasyn_oversampling(self,X, y, sampling_strategy='auto', random_state=42):
        self.LogCreator.print_and_write_log(f"Start ADASYN oversampling\n"
                                            f"Data before ADASYN_oversampling: {pd.Series(y).value_counts()}")
        adasyn_start = time.time()
        class_counts = pd.Series(y).value_counts()
        min_class_size = class_counts.min()
        n_neighbors = min(min_class_size - 1, 5)
        self.LogCreator.print_and_write_log(f"n_neigh = {n_neighbors}\n")
        if n_neighbors < 1:
            self.LogCreator.print_and_write_log(
                "ADASYN cannot be applied due to insufficient class samples. Skipping oversampling."
            )
            return X, y
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, n_neighbors=n_neighbors)
        try:
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
        except ValueError as e:
            self.LogCreator.print_and_write_log(
                f"ADASYN failed with error: {str(e)}. Returning original data."
            )
            return X, y
        adasyn_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End ADASYN oversampling\n"
            f"Data after ADASYN_oversampling: {pd.Series(y_resampled).value_counts()}\n"
            f"Time to ADASYN oversampling: {self.LogCreator.count_time(adasyn_start, adasyn_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}"
        )

        return X_resampled, y_resampled

    def smote_tomek(self,X, y, sampling_strategy='auto', random_state=42):
        self.LogCreator.print_and_write_log(f"Start SMOTE_TOMEK \n"
                                            f"Data before SMOTE_TOMEK: {pd.Series(y).value_counts()}")
        smote_tomek_start = time.time()
        smote_tomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        smote_tomek_end = time.time()
        self.LogCreator.print_and_write_log(f"End SMOTE_TOMEK \n"
                                            f"Data after SMOTE_TOMEK: {pd.Series(y_resampled).value_counts()}\n"
                                            f"Time to SMOTE_TOMEK: {self.LogCreator.count_time(smote_tomek_start, smote_tomek_end):.2f} s.\n"
                                            f"{self.LogCreator.string_spit_stars}")
        return X_resampled, y_resampled
    def smote_enn(self,X, y, sampling_strategy='auto', random_state=42):
        self.LogCreator.print_and_write_log(f"Start SMOTE_ENN \n"
                                            f"Data before SMOTE_ENN: {pd.Series(y).value_counts()}")
        smote_enn_start = time.time()
        smote_enn = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        smote_enn_end = time.time()
        self.LogCreator.print_and_write_log(f"End SMOTE_ENN \n"
                                            f"Data after SMOTE_ENN: {pd.Series(y_resampled).value_counts()}\n"
                                            f"Time to SMOTE_ENN: {self.LogCreator.count_time(smote_enn_start, smote_enn_end):.2f} s.\n"
                                            f"{self.LogCreator.string_spit_stars}")
        return X_resampled, y_resampled
    def random_under(self, X,y, sampling_strategy='auto', random_state=42):
        self.LogCreator.print_and_write_log(f"Start RandomUnder \n"
                                            f"Data before RandomUnder: {pd.Series(y).value_counts()}")
        RandomUnder_start = time.time()
        RandomUnder = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = RandomUnder.fit_resample(X, y)
        RandomUnder_end = time.time()
        self.LogCreator.print_and_write_log(f"End RandomUnder \n"
                                            f"Data after RandomUnder: {pd.Series(y_resampled).value_counts()}\n"
                                            f"Time to RandomUnder: {self.LogCreator.count_time(RandomUnder_start, RandomUnder_end):.2f} s.\n"
                                            f"{self.LogCreator.string_spit_stars}")
        return X_resampled, y_resampled