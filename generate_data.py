import random
import mmh3
import numpy as np
import pandas as pd
import os
from typing import List

# Constants
NUCLEOTIDES = ['A', 'C', 'G', 'T']
K = 31  # Length of k-mer
HASH_SEED = 42
VARIANTS_PER_KMER = 10
SAVE_DIR = '.'

def random_kmer(k: int) -> str:
    return ''.join(random.choices(NUCLEOTIDES, k=k))

def mutate_kmer(kmer: str, max_mutations: int = 5) -> str:
    num_mut = random.randint(1, max_mutations)
    kmer = list(kmer)
    pos = random.sample(range(len(kmer)), num_mut)
    for i in pos:
        kmer[i] = random.choice([n for n in NUCLEOTIDES if n != kmer[i]])
    return ''.join(kmer)

def encode_kmer(kmer: str) -> List[int]:
    onehot = {'A': [1, 0, 0, 0],
              'C': [0, 1, 0, 0],
              'G': [0, 0, 1, 0],
              'T': [0, 0, 0, 1]}
    return [bit for base in kmer for bit in onehot[base]]

def generate_dataset(n_base_kmers: int, k: int, seed: int = 0) -> pd.DataFrame:
    random.seed(seed)
    kmers = [random_kmer(k) for _ in range(n_base_kmers)]

    rows = []

    for kmer in kmers:
        hash_val = mmh3.hash(kmer, seed=HASH_SEED, signed=False) / 2**32
        encoded = encode_kmer(kmer)
        rows.append(encoded + [hash_val])

        for _ in range(VARIANTS_PER_KMER):
            mutated = mutate_kmer(kmer)
            rows.append(encode_kmer(mutated) + [hash_val])

    colnames = [f'x_{i}' for i in range(4 * k)] + ['target']
    return pd.DataFrame(rows, columns=colnames)

def save_csv(name: str, df: pd.DataFrame, save_dir: str = SAVE_DIR):
    path = os.path.join(save_dir, name + ".csv")
    df.to_csv(path, index=False)
    print(f"Saved {name}.csv with shape {df.shape}")

def main():
    print("Generating datasets...")

    df_train = generate_dataset(n_base_kmers=10_000, k=K, seed=0)
    df_valid = generate_dataset(n_base_kmers=1_000, k=K, seed=1)
    df_test  = generate_dataset(n_base_kmers=5_000,  k=K, seed=2)

    save_csv("train", df_train)
    save_csv("valid", df_valid)
    save_csv("test",  df_test)

if __name__ == "__main__":
    main()
