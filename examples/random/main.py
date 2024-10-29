"""This is an example submission that just generates random predictions."""

from pathlib import Path

import librosa
from loguru import logger
import numpy as np
import pandas as pd

DATA_DIR = Path("data/")


def main():
    # load the two csvs in the data directory
    df = pd.read_csv(DATA_DIR / "submission_format.csv", index_col="filename")
    metadata = pd.read_csv(DATA_DIR / "test_metadata.csv", index_col="filename")

    # set random state for a reproducible submission since we're generating random probabilities
    rng = np.random.RandomState(99)

    # iterate over audio files
    scores = []
    for file in df.index:
        logger.info(f"Loading {file}")
        audio, sr = librosa.load(DATA_DIR / file)

        # since this is a dummy submission, just assign a random number between 0 and 1
        scores.append(rng.random())

    # write the scores to score column in the submission format
    df["score"] = scores

    # write out predictions to submission.csv in the main directory
    logger.info("Writing out submission.csv")
    df.to_csv("submission.csv")


if __name__ == "__main__":
    main()
