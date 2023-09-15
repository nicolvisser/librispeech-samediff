import pandas as pd
import numpy as np


def print_summary_row(df, subset):
    words = words = df.word.to_numpy()
    speakers = df.speaker_id.to_numpy()

    SW = words[:, np.newaxis] == words[np.newaxis, :]
    SP = speakers[:, np.newaxis] == speakers[np.newaxis, :]

    SWSP = np.logical_and(SW, SP)
    SWDP = np.logical_and(SW, ~SP)
    DWSP = np.logical_and(~SW, SP)
    DWDP = np.logical_and(~SW, ~SP)

    upper_tri = np.triu_indices(len(words), k=1)

    n_words = len(words)
    n_pairs = np.ones_like(SW, dtype=int)[upper_tri].sum()
    n_SWSP = SWSP[upper_tri].sum()
    n_SWDP = SWDP[upper_tri].sum()
    n_DWSP = DWSP[upper_tri].sum()
    n_DWDP = DWDP[upper_tri].sum()

    print(f"| {subset} | {n_words:,} | {n_pairs:,} | {n_SWSP:,} | {n_SWDP:,} | {n_DWSP:,} | {n_DWDP:,} |")


if __name__ == "__main__":
    print("| Subset | Word Count | Word Pairs | SWSP | SWDP | DWSP | DWDP |")
    print("| --- | --- | --- | --- | --- | --- | --- |")

    for subset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        df = pd.read_csv(f"librispeech_samediff/data/{subset}.csv")
        print_summary_row(df, subset)

    df = pd.concat(
        [
            pd.read_csv(f"librispeech_samediff/data/dev-clean.csv"),
            pd.read_csv(f"librispeech_samediff/data/dev-other.csv"),
        ]
    )

    print_summary_row(df, "dev")

    df = pd.concat(
        [
            pd.read_csv(f"librispeech_samediff/data/test-clean.csv"),
            pd.read_csv(f"librispeech_samediff/data/test-other.csv"),
        ]
    )

    print_summary_row(df, "test")
