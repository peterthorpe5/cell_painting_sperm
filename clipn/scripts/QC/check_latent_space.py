#!/usr/bin/env python3
"""
Check if CLIPn latent vectors are unit-normalised.

Reads a latent TSV file and reports min, max, mean, and std of L2 norms across vectors.

Also plots a histogram of the norms if `--plot` is passed.

Usage:
    python check_latent_vector_norms.py --input latent.tsv [--plot]

"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def check_norms(file_path: str, plot: bool = False) -> None:
    """
    Load latent file, compute L2 norms from specific embedding columns, and optionally plot histogram.

    Parameters:
        file_path (str): Path to the latent space TSV file.
        plot (bool): Whether to plot a histogram of norms.
    """
    df = pd.read_csv(file_path, sep="\t")

    # Explicitly select latent columns by name (as strings)
    latent_cols = [str(i) for i in range(20)]
    missing = [col for col in latent_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing expected latent columns: {missing}")

    features = df[latent_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with any NaNs in the embedding
    features = features.dropna()
    norms = np.linalg.norm(features.values, axis=1)

    print(f"Checked {len(norms)} latent vectors.")
    print(f"Min norm:  {norms.min():.6f}")
    print(f"Max norm:  {norms.max():.6f}")
    print(f"Mean norm: {norms.mean():.6f}")
    print(f"Std dev:   {norms.std():.6f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.hist(norms, bins=30, color="skyblue", edgecolor="black")
        plt.axvline(1.0, color="red", linestyle="--", label="Norm = 1.0")
        plt.xlabel("L2 Norm")
        plt.ylabel("Frequency")
        plt.title("Distribution of Latent Vector Norms")
        plt.legend()
        plt.tight_layout()
        plt.savefig("latent_vector_norm_histogram.pdf", dpi=300)
        plt.close()
        print("Histogram saved to latent_vector_norm_histogram.pdf")



def main():
    parser = argparse.ArgumentParser(description="Check norm distribution of CLIPn latent vectors.")
    parser.add_argument("--input", required=True, help="Path to latent TSV file")
    parser.add_argument("--plot", action="store_true", help="Plot histogram of norms")
    args = parser.parse_args()
    check_norms(args.input, plot=args.plot)





if __name__ == "__main__":
    main()
