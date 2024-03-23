# UniUK Sentiment Dashboard

This repository contains a data visualization dashboard for analyzing and visualizing data from the subreddit r/UniUK from February 2016 (from the conceptualization of the subreddit) to December 2023. The data has been collected through an open-source project named PushShift and includes a vast number of posts and comments that offer insights into university life in the UK.

## Data Preprocessing

The data preprocessing steps, including text refinement using the NLTK library, sentiment analysis using VaderSentiment, and topic modeling using BerTopic, are not covered in this repository. The preprocessing code can be found at [https://www.kaggle.com/code/sgjustino/uniuk-topic-modeling-with-bertopic?scriptVersionId=168342984](https://www.kaggle.com/code/sgjustino/uniuk-topic-modeling-with-bertopic?scriptVersionId=168342984).

## Dashboard

The visualization part of the project showcases the identified themes and their popularity over time, as well as the sentiment (positive, neutral, negative) associated with each theme. It also includes a data table to explore the actual data. The dashboard is built using Dash (Plotly).

## Repository Structure

- `app.py`: The main Python script that contains the Dash application code for the visualization dashboard.
- `requirements.txt`: A file listing the required Python packages and their versions to run the application.
- `data/uniuk_sentiment_data.csv`: The preprocessed dataset used for visualization.
- `assets/styles.css`: A CSS file containing custom styles for the dashboard.

## App.py Components

The `app.py` script contains the following main components:

1. **Data Processing**: This section loads the data from the CSV file, removes error data, converts the date correctly, merges Topic 1 to outliers, and shifts topic numbers sequentially.

2. **Dashboard App**: This section initializes the Dash application and sets up the overall layout of the dashboard.

3. **Topic Frequency Page**: This section defines the layout and components for the topic frequency analysis page, including a range slider for selecting the range of topics and tabs for displaying absolute and normalized frequencies.

4. **Sentiment Analysis Page**: This section defines the layout and components for the sentiment analysis page, including a dropdown for selecting the topic of interest and tabs for displaying absolute and normalized frequencies.

5. **Topic Data Page**: This section defines the layout and components for the topic data page, including a dropdown for selecting the topic of interest and a range slider for selecting the range of years.

6. **Callbacks**: This section contains the callbacks that update the visualizations and data tables based on user interactions with the dashboard components.

## Usage

1. Install the required Python packages listed in `requirements.txt` by running the following command:
   ```
   pip install -r requirements.txt
   ```

2. Run `app.py` to start the Dash application:
   ```
   python app.py
   ```

3a. Access the dashboard through localhost:
   ```
   http://127.0.0.1:8050/
   ```

3b. Access the dashboard through the provided URL in your web browser:
   ```
   http://127.0.0.1:8050/
   ```

## Data Source

The original data source can be found at [https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4](https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4).

## Libraries/Tools Used

- NLTK: [https://github.com/nltk/nltk](https://github.com/nltk/nltk)
- BERTopic: [https://github.com/MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)
- VaderSentiment: [https://github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment)
- Dash (Plotly): [https://github.com/plotly/dash] (https://github.com/plotly/dash)
- Dash Opioid Epidemic Demo (inspiration for sliders): [https://github.com/plotly/dash-opioid-epidemic-demo](https://github.com/plotly/dash-opioid-epidemic-demo)
- Dash Manufacture SPC Dashboard (inspiration for tabs): [https://github.com/dkrizman/dash-manufacture-spc-dashboard](https://github.com/dkrizman/dash-manufacture-spc-dashboard)

## References

- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
- Solatorio, A. V. (2024). GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning. arXiv preprint arXiv:2402.16829. [https://arxiv.org/abs/2402.16829](https://arxiv.org/abs/2402.16829)
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.

### End