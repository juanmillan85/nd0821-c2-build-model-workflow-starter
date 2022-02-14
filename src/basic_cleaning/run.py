#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = loading_data(artifact_local_path)
    df = drop_price_outliers(args, df)

    df = drop_lat_lon_outliers(df)

    convert_lastreview_datetime(df)

    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


def convert_lastreview_datetime(df):
    df["last_review"] = pd.to_datetime(df["last_review"])


def drop_lat_lon_outliers(df):
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    return df[idx].copy()


def drop_price_outliers(args, df):
    logger.info("Drop Outliers")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    return df


def loading_data(artifact_local_path):
    logger.info("Loading Dataframe")
    df = pd.read_csv(artifact_local_path)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", type=str, help="Name for the input artifact", required=True
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Name for the ouput artifact", required=True
    )

    parser.add_argument(
        "--output_type", type=str, help="Type for the output artifact", required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", type=float, help="Minimum price for outlier range", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="Maximum price for outlier range", required=True
    )

    args = parser.parse_args()

    go(args)
