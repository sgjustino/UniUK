# UniUK Sentiment Dashboard

This repository contains a data visualization dashboard for analyzing and visualizing data from the subreddit r/UniUK from February 2016 (the inception of Subreddit r/UniUK) to December 2023. The data has been collected through an open-source project named PushShift and includes a vast number of posts and comments that offer insights into university life in the UK.

## [Access the webpage](https://uniuk.pythonanywhere.com/)


## Navigating the Dashboard

The dashboard is built using Dash (Plotly) with the following components:

* Background Page: Introduces the study motivation and research question, guides users on exploring the dashboard findings and provide additional details like data source, preprocessing steps and references.
* Topic Frequency Page: Allows users to view the frequency of selected topics over time, either as absolute counts or normalized percentages, to identify popular topics and trends over time.
* Sentiment Analysis Page: Enables users to analyze sentiment trends for a specific topic over time, using absolute counts or normalized percentages views, to understand the emotional tone of discussions.
* Topic Data Page: Provides a table view of the individual posts for a selected topic and year range, with sentiment indicated by cell color, allowing users to explore specific discussions.
* Interpretation Page: Demonstrates how to use the dashboard to examine the research question through an example analysis of a specific topic, showcasing insights and conclusions.
    
## Data Source and Preprocessing

The data, spanning from February 2016 (the inception of Subreddit r/UniUK) to December 2023, was obtained from academic torrents hosted online and collected by an open-source project called Pushshift. To prepare the data for analysis and answer the research question, several pre-processing steps and modeling were undertaken. First, the data was cleaned using custom stopwords and the NLTK library to remove irrelevant information and improve the quality of the dataset. Next, sentiment analysis was performed using VaderSentiment to determine the polarity (positive, neutral, and negative) of each post. Finally, topic modeling was conducted using BerTopic to identify and categorize the main themes within the data. 

To focus on the visualisation aspects, the detailed data modeling steps are not covered in this project repository. However, the modeling process are shared in the accompanying Kaggle notebook, providing a reproducible account of the data analysis. 

- [Link to Data Source](https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10)
- [Link to Modeling Notebook](https://www.kaggle.com/code/sgjustino/uniuk-topic-modeling-with-bertopic?scriptVersionId=168342984)

## Meta-Data for Processed Data

* body: The text content within a post. It encompasses both initial submissions (which start discussion threads) and subsequent comments. By analyzing these elements collectively, we treat them as a unified set of social media posts for examination.
* created_utc: The timestamp of when the post was created.
sentiment: The sentiment of the post as determined by VaderSentiment (positive, neutral, or negative).
* processed_text: The processed content of the post using custom stopwords and NLTK library.
* Topic: The topic number that the post belongs to as determined by BerTopic. Topic 74 refers to the outliers not classified into any specific topic.
* Topic_Label: The descriptive label assigned to each topic, derived from the four most representative words identified through a class-based Term Frequency-Inverse Document Frequency analysis of the topic's content (Grootendorst, 2022).

## Repository Structure

- `app.py`: The main Python script that contains the Dash application code for the visualization dashboard.
- `dockerfile`: Dockerfile to build docker image for the repository.
- `interpretation.py`: The Python script that generate the htmls used for interpretation page.
- `requirements.txt`: A file listing the required Python packages and their versions to run the application.
- `assets/styles.css`: A CSS file containing custom styles for the dashboard.
- `assets/future_direction.gif`: A gif file showcasing an example of AI-powered visualisations.
- `data/uniuk_sentiment_data.csv`: The preprocessed dataset used for visualization.
- `fig/`: Folder containing visualisations saved and used in interpretation page (interpretation.py).
- `tests/test_app.py`: A python script for unit tests and data checks.
- `LICENSE.md`: The MIT License.

## App.py Components

The `app.py` script contains the following main components:

1. **Data Processing**: This section loads the data from the CSV file, removes error data, converts the date correctly, merges Topic 1 to outliers, and shifts topic numbers sequentially.

2. **Dashboard App**: This section initializes the Dash application and sets up the overall layout of the dashboard.

3. **Background Page**: This section defines the layout and components for the background page, including the data table to explain the meta-data.

4. **Topic Frequency Page**: This section defines the layout and components for the topic frequency analysis page, including a range slider for selecting the range of topics and tabs for displaying absolute and normalized frequencies.

5. **Sentiment Analysis Page**: This section defines the layout and components for the sentiment analysis page, including a dropdown for selecting the topic of interest and tabs for displaying absolute and normalized frequencies.

6. **Topic Data Page**: This section defines the layout and components for the topic data page, including a dropdown for selecting the topic of interest and a range slider for selecting the range of years.

7. **Interpretation Page**: This section defines the layout and components for the interpretation page, including the generation of the 3 htmls from interpretation.py and the example gif.

8. **Callbacks**: This section contains the callbacks that update the visualizations and data tables based on user interactions with the dashboard components.

## Running Locally

1. Install the required Python packages listed in `requirements.txt` by running the following command:
   ```
   pip install -r requirements.txt
   ```

2. Run `app.py` to start the Dash application:
   ```
   python app.py
   ```

3. Access the dashboard through localhost:
   ```
   http://127.0.0.1:8050/
   ```

## Built With

- [Pre-processing with NLTK](https://github.com/nltk/nltk)
- [Topic Modeling with BERTopic](https://github.com/MaartenGr/BERTopic)
- [Sentiment Classification with VADER](https://github.com/cjhutto/vaderSentiment)
- [Dashboard Development with Dash (Plotly)](https://github.com/plotly/dash)
- [Inspiration for range sliders from Dash Opioid Epidemic Demo](https://github.com/plotly/dash-opioid-epidemic-demo)
- [Inspiration for tabs from Dash Manufacture SPC Dashboard](https://github.com/dkrizman/dash-manufacture-spc-dashboard)
- ChatGPT4 and Claude 3 Opus were utilised for code development and bug fixing.

## References

- Al-Natour, S., & Turetken, O. (2020). A comparative assessment of sentiment analysis and star ratings for consumer reviews. International Journal of Information Management, 54, 102132.
- Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020, May). The pushshift reddit dataset. In Proceedings of the international AAAI conference on web and social media (Vol. 14, pp. 830-839).
- Biswas, A. (2023, October 17). AI-powered data visualizations: Introducing an app to generate charts using only a single prompt and OpenAI large language models. Databutton. https://medium.com/databutton/ai-powered-data-visualization-134e89d82d99
- Briggs, A. R., Clark, J., & Hall, I. (2012). Building bridges: understanding student transition to university. Quality in higher education, 18(1), 3-21.
- Gagné, T., Schoon, I., McMunn, A., & Sacker, A. (2021). Mental distress among young adults in Great Britain: long-term trends and early changes during the COVID-19 pandemic. Social Psychiatry and Psychiatric Epidemiology, 1-12.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
- Grootendorst, M. (2023, August 22). Topic modeling with Llama 2: Create easily interpretable topics with Large Language Models. Towards Data Science. https://towardsdatascience.com/topic-modeling-with-llama-2-85177d01e174
- Guo, Y., Ge, Y., Yang, Y. C., Al-Garadi, M. A., & Sarker, A. (2022). Comparison of pretraining models and strategies for health-related social media text classification. In Healthcare (Vol. 10, No. 8, p. 1478). MDPI.
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
- Reddy, K. J., Menon, K. R., & Thattil, A. (2018). Academic stress and its sources among university students. Biomedical and pharmacology journal, 11(1), 531-537.
- Samaras, L., García-Barriocanal, E., & Sicilia, M. A. (2023). Sentiment analysis of COVID-19 cases in Greece using Twitter data. Expert Systems with Applications, 230, 120577.
- Solatorio, A. V. (2024). GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning. arXiv preprint arXiv:2402.16829.
- Souza, F. D., & Filho, J. B. D. O. E. S. (2022). BERT for sentiment analysis: pre-trained and fine-tuned alternatives. In International Conference on Computational Processing of the Portuguese Language (pp. 209-218). Cham: Springer International Publishing.
- Winstone, L., Mars, B., Haworth, C. M., & Kidger, J. (2021). Social media use and social connectedness among adolescents in the United Kingdom: a qualitative exploration of displacement and stimulation. BMC public health, 21, 1-15.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

### End