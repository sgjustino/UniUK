# Imports
import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import sentiment_data, topic_label_df, update_topic_frequency_graph, update_sentiment_analysis_graph


# Test the data preprocessing steps to ensure error data is removed, 'created_utc' column is converted to datetime format, and topic numbers are updated correctly
def test_data_preprocessing():
    # Verify that error data is removed correctly
    assert sentiment_data["Topic_Label"].notna().all()

    # Ensure that the 'created_utc' column is converted to the correct datetime format
    assert sentiment_data["created_utc"].dtype == "datetime64[ns]"

    # Test if the topic numbers are updated correctly (Topic 99 to Topic 75, Topic 1 to Topic 75, and decrementing all topic numbers by 1)
    assert sentiment_data["Topic"].max() == 74  # Maximum topic number after preprocessing is 74
    assert "Topic 74: Outliers" in sentiment_data["Topic_Label"].unique()

# Test the dropdown options generation to verify the options are generated correctly based on unique topic labels and sorted based on topic numbers
def test_dropdown_options_generation():
    # Verify that the dropdown options are generated correctly based on the unique topic labels
    dropdown_options = [{'label': row['Topic_Label'], 'value': row['Topic_Label']} for _, row in topic_label_df.iterrows()]
    assert len(dropdown_options) == len(topic_label_df)
    assert all(option["label"] == topic_label_df.loc[topic_label_df["Topic_Label"] == option["value"], "Topic_Label"].iloc[0] for option in dropdown_options)

    # Ensure that the options are sorted based on the topic numbers
    topic_numbers = [int(option["label"].split(":")[0].split(" ")[1]) for option in dropdown_options]
    assert topic_numbers == sorted(topic_numbers)

# Test the topic frequency normalization to ensure the frequencies are normalized correctly
def test_topic_frequency_normalization():
    # Retrieve the normalized frequency data from the update_topic_frequency_graph function
    selected_range = [1, 10]
    frequency_type = 'normalized'
    fig = update_topic_frequency_graph(selected_range, frequency_type)

    # Extract the normalized frequency data from the figure traces
    normalized_frequencies = [trace["y"] for trace in fig["data"] if "Normalized" in trace["name"]]

    # Ensure that the normalized frequencies sum up to 100% for each quarter
    assert all(abs(sum(freqs) - 100) < 1e-6 for freqs in zip(*normalized_frequencies))

# Test the consistency between Topic and Topic_Label columns
def test_topic_label_consistency():
    # Ensure that the Topic and Topic_Label columns have the same number of unique values
    assert len(sentiment_data["Topic"].unique()) == len(sentiment_data["Topic_Label"].unique())

    # Verify that each Topic value has a corresponding Topic_Label
    topic_labels = sentiment_data.groupby("Topic")["Topic_Label"].unique().apply(list)
    assert all(len(labels) == 1 for labels in topic_labels)

# Test the sentiment column to ensure it contains only valid sentiment labels
def test_sentiment_labels():
    # Define the valid sentiment labels
    valid_sentiments = ["positive", "neutral", "negative"]

    # Verify that the sentiment column contains only valid sentiment labels
    assert sentiment_data["sentiment"].isin(valid_sentiments).all()