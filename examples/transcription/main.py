"""This is an example submission that uses a pretrained model to generate transcriptions.
Note: for this submission to work, you must download the whisper model to the assets/ dir first."""

import string
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import torch
import whisper

DATA_DIR = Path("data/")


def download_whisper_model(download_root="assets"):
    """Code to download model locally so we can include it in our submission"""
    whisper.load_model("turbo", download_root=download_root)


def clean_column(col: pd.Series):
    return col.str.lower().str.strip().replace(f"[{string.punctuation}]", "", regex=True)


def main():
    # load the metadata that has the expected text for each audio file
    df = pd.read_csv(DATA_DIR / "test_metadata.csv", index_col="filename")

    # load whisper model and put on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("assets/large-v3-turbo.pt").to(device)

    # iterate over audio files and get transcribed text
    transcribed_texts = []
    for file in df.index:
        logger.info(f"Transcribing {file}")
        # set temperature at 0 for reproducible results
        result = model.transcribe(str(DATA_DIR / file), language="english", temperature=0)
        transcribed_texts.append(result["text"])

    df["transcribed_text"] = transcribed_texts

    # clean columns to avoid false mismatches
    df["expected_text"] = clean_column(df.expected_text)
    df["transcribed_text"] = clean_column(df.transcribed_text)

    # score = 1 if transcribed text matches expected text
    # score = 0.5 if transcription doesn't match; avoids penalizing confident but wrong
    df["score"] = np.where(df.transcribed_text == df.expected_text, 1.0, 0.5)

    # ensure index matches submission format
    sub_format = pd.read_csv(DATA_DIR / "submission_format.csv", index_col="filename")
    preds = df[["score"]].loc[sub_format.index]

    # write out predictions to submission.csv in the main directory
    logger.info("Writing out submission.csv")
    preds.to_csv("submission.csv")


if __name__ == "__main__":
    main()
