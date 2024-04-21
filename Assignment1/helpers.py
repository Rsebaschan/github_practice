# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset,
        delimiter=",",
        skip_header=1,
        usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1},
    )
    # Convert to metric system
    height *= 0.025
    weight *= 0.454
    return height, weight, gender


def sample_data(y, x, seed, size_samples): #데이터셋으로부터 일부 샘플을 추출하는 역할
    """sample from dataset."""
    np.random.seed(seed)  #난수 생성기 seed를 생성 #seed는 함수의 인자중 하나다. 난수 생성의 시작점을 결정
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]  #결과적으로, 이 함수는 전체 데이터셋에서 무작위로 선택된 샘플의 부분 집합을 반환


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)    # 원래 x는 데이터 값
    x = x - mean_x             # 원래 x - 평균 x  = 바뀐 x
    std_x = np.std(x, axis=0) # variance 분산 sigma^2 == sigma**2 #std = standard deviation  표준편차  sigma   
    x = x / std_x             # 바뀐 x / x의 표준편차 = 표준화(standardize)z    # 표준화 공식 z = (원래x -평균x) / 표준편차      
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x): # 바뀐 x, 평균 , 표준편차 를 이용해서 원래 x찾기
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x  # 원래 x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
