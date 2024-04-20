# Imports
import pandas as pd 
import os
import plotly.graph_objs as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from interpretation import generate_topic_frequency_html, generate_sentiment_analysis_html, generate_topic_data_table
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
This app contains the following main components:

* Data Processing: This section loads the data from the CSV file, removes error 
  data, converts the date correctly, merges Topic 1 to outliers, and shifts 
  topic numbers sequentially.

* Dashboard App: This section initializes the Dash application and sets up the 
  overall layout of the dashboard.

* Background Page: This section defines the layout and components for the 
  background page, including the data table to explain the meta-data.

* Topic Frequency Page: This section defines the layout and components for the 
  topic frequency analysis page, including a range slider for selecting the 
  range of topics and tabs for displaying absolute and normalized frequencies.

* Sentiment Analysis Page: This section defines the layout and components for 
  the sentiment analysis page, including a dropdown for selecting the topic of 
  interest and tabs for displaying absolute and normalized frequencies.

* Topic Data Page: This section defines the layout and components for the topic 
  data page, including a dropdown for selecting the topic of interest and a 
  range slider for selecting the range of years.

* Interpretation Page: This section defines the layout and components for the 
  interpretation page, including the generation of the 3 htmls from 
  interpretation.py and the example gif.

* Callbacks: This section contains the callbacks that update the visualizations 
  and data tables based on user interactions with the dashboard components.
'''



#################
# Data Processing
#################

# Load Data, remove error data and convert date correctly
# Build the path to the file
file_path = os.path.join(os.getcwd(), 'data', 'uniuk_sentiment_data.csv')

# Read the CSV file into a DataFrame
sentiment_data = pd.read_csv(file_path)

# Remove error data
sentiment_data = sentiment_data[sentiment_data['Topic_Label'].notna()]

#convert the 'created_utc' column to datetime format
sentiment_data['created_utc'] = pd.to_datetime(sentiment_data['created_utc'])

'''
(1) Update Topic and Topic_Label for rows with Topic 99 to Topic 75 to prevent range break for range slider
(2) Merge Topic 1 to Outliers as Topic 1 (Thank, Thanks, Comment, Post) is not useful
(3) Decrement all Topic and Topic_Label values by 1 (with the removal of Topic 1)

Note: this step was interatively taken during the data sense-making
'''

# Update Topic 99 to 75
sentiment_data.loc[sentiment_data['Topic'] == 99, 'Topic'] = 75

# Update Topic 1 to Outliers
sentiment_data.loc[sentiment_data['Topic'] == 1, 'Topic'] = 75
sentiment_data.loc[sentiment_data['Topic'] == 75, 'Topic_Label'] = 'Topic 75: Outliers'

# Shift all topics down by 1
sentiment_data['Topic'] -= 1
sentiment_data['Topic_Label'] = sentiment_data['Topic_Label'].apply(
    lambda x: f"Topic {int(x.split(':')[0].split(' ')[1]) - 1}:{x.split(':')[1]}")

'''
Process Topics for dropdown list usage with Topic Number ordering
'''

# Create a DataFrame with unique Topic_Label and their corresponding Topic values
topic_label_df = sentiment_data[['Topic', 'Topic_Label']].drop_duplicates()

# Sort the DataFrame by Topic values
topic_label_df = topic_label_df.sort_values('Topic')

# Create the options for the dropdown
dropdown_options = [{'label': row['Topic_Label'], 'value': row['Topic_Label']} for _, row in topic_label_df.iterrows()]



#################
# Dashboard App 
#################

# Main App server with styles.css
app = dash.Dash(__name__, external_stylesheets=["assets/styles.css"], suppress_callback_exceptions=True)
server = app.server
pd.options.mode.chained_assignment = None

# Set topic range (allow update of data and topics)
try:
    topic_max = int(sentiment_data['Topic'].max())
except ValueError:
    topic_max = 10



#################
# App Layout
#################
    
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Title Row
    html.Div(
        # Main Title
        className='main-title',
        children=[
            html.Div(
                children=[
                    html.H2('How do the perspectives of UK university students, as expressed on Reddit, evolve over time?'),
                    html.P(
                        'Examining key themes and sentiment trends from r/UniUK posts (2016-2023) to understand the changing perspectives and emotional dynamics of UK university students.',
                        style={'fontSize': '14px', 'color': '#efefef'}
                    ),
                ]
            )
        ]),

    # Tabs Row (Switch between pages)
    html.Div(className='row', children=[
        html.Div(
            className='twelve columns',
            children=[
                dcc.Tabs(
                    id="app-tabs",
                    value="tab0",
                    className="custom-tabs",
                    children=[
                        dcc.Tab(
                            id="Background-tab",
                            label="Background",
                            value="tab0",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="Topic-Frequency-tab",
                            label="Topic Analysis",
                            value="tab1",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="Sentiment-Analysis-tab",
                            label="Sentiment Analysis",
                            value="tab2",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="Topic-Data-tab",
                            label="Topic Data",
                            value="tab3",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="Interpretation-tab",
                            label="Interpretation",
                            value="tab4",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                    ],
                )
            ]
        ),
    ]),

    # Content Row (individual pages)
    html.Div(className='row', children=[
        html.Div(
            className='twelve columns',
            children=[html.Div(id='page-content')]        )
    ])
])



#################
# Background Page
#################

background_page = html.Div([
    # Background Info and Motivation
    html.H2("Background", className="title"),    
    html.Hr(),
    dcc.Markdown('''
    Traditional research studies, often relying on methodologies such as surveys, interviews, or focus groups, are typically constrained by the scale of recruitment and tend to capture student perspectives in more structured, formal environments (e.g. Briggs, Clark & Hall, 2012; Reddy, Menon & Thattil, 2018).
    In contrast, social media platforms are rich with spontaneous, unfiltered discussions about university life, offering a wider and more authentic range of student viewpoints.
    Analyzing these social media conversations can reveal subtle and candid insights into the university experience - insights that students might be reluctant to share in the more controlled settings of traditional research methods.
    Besides that, this approach provides access to a larger volume of data points, capturing a diverse array of student voices and experiences.
    Recognizing the rich insights that social media conversations offer, we thus turn our attention to the research question:
    '''),
    html.Hr(),
    # Explanation on how we will address the RQ
    html.H2("How do the perspectives of UK university students, as expressed on Reddit, evolve over time?", className="title"), 
    html.Hr(),
    dcc.Markdown('''
    To address this, we examined the social media data from the subreddit r/UniUK, a hub for candid UK university student discussions. 
    Our analysis of the numerous posts and comments within this community has led to the identification of key themes and the sentiment trends associated with them over time. 
    Users can navigate the interactive dashboard to explore specific areas of interest, thereby gaining a nuanced understanding of how the sentiments of UK university students on Reddit have evolve over time. 
    This tool not only aids in answering our research question but also provides a foundation for further research and informed decision-making within the realm of higher education.
    '''),
    html.Hr(),

    # How to navigate the dashboard
    html.H2("Navigating the Dashboard", className="title"),
    html.Hr(),
    dcc.Markdown('''
    The dashboard is built using Dash (Plotly) with the following components:
    '''),
    html.Li("Background Page: Introduces the study motivation and research question, guides users on exploring the dashboard findings and provide additional details like data source, preprocessing steps and references."),
    html.Li("Topic Frequency Page: Allows users to view the frequency of selected topics over time, either as absolute counts or normalized percentages, to identify popular topics and trends over time."),
    html.Li("Sentiment Analysis Page: Enables users to analyze sentiment trends for a specific topic over time, using absolute counts or normalized percentages views, to understand the emotional tone of discussions."),
    html.Li("Topic Data Page: Provides a table view of the individual posts for a selected topic and year range, with sentiment indicated by cell color, allowing users to explore specific discussions."),
    html.Li("Interpretation Page: Demonstrates how to use the dashboard to examine the research question through an example analysis of a specific topic, showcasing insights and conclusions."),
    html.Li(html.A("Find out more at the github repository", href="https://github.com/sgjustino/UniUK", target="_blank")),
    html.Hr(),

    # Data Source and Pre-processing steps taken
    html.H2("Data Source and Preprocessing", className="title"),
    html.Hr(),
    dcc.Markdown('''
    The data, spanning from February 2016 (the inception of Subreddit r/UniUK) to December 2023, was obtained from academic torrents hosted online and collected by an open-source project called Pushshift. 
    To prepare the data for analysis and answer the research question, several pre-processing steps and modeling were undertaken. 
    First, the data was cleaned using custom stopwords and the NLTK library to remove irrelevant information and improve the quality of the dataset. 
    Next, sentiment analysis was performed using VaderSentiment to determine the polarity (positive, neutral, and negative) of each post.
    Finally, topic modeling was conducted using BerTopic to identify and categorize the main themes within the data.
    '''),
    dcc.Markdown('''
    To focus on the visualisation aspects, the detailed data modeling steps are not covered in this project repository. 
    However, the modeling process are shared in the accompanying Kaggle notebook, providing a reproducible account of the data analysis.
    '''),
    html.Li(html.A("Link to Data Source", href="https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10", target="_blank")),
    html.Li(html.A("Link to Modeling Notebook", href="https://www.kaggle.com/code/sgjustino/uniuk-topic-modeling-with-bertopic?scriptVersionId=168342984", target="_blank")),
    html.Hr(),

    # Meta-data or Codebook
    html.H2("Meta-Data for Processed Data", className="title"),
    html.Hr(),
    # Data Table for the Processed data used in Visualisation
    html.Div([
        dash_table.DataTable(
            data=sentiment_data.head().to_dict('records'),
            columns=[{"name": i, "id": i} for i in sentiment_data.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': '#26282A',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'fontFamily': 'Lato, sans-serif',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'backgroundColor': 'black',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'fontFamily': 'Lato, sans-serif',
            }
        ),

        # Meta-data explanation
        html.Ul([
            html.Li("body: The text content within a post. It encompasses both initial submissions (which start discussion threads) and subsequent comments. By analyzing these elements collectively, we treat them as a unified set of social media posts for examination."),
            html.Li("created_utc: The timestamp of when the post was created."),
            html.Li("sentiment: The sentiment of the post as determined by VaderSentiment (positive, neutral, or negative)."),
            html.Li("processed_text: The processed content of the post using custom stopwords and NLTK library."),
            html.Li("Topic: The topic number that the post belongs to as determined by BerTopic. Topic 74 refers to the outliers not classified into any specific topic."),
            html.Li("Topic_Label: The descriptive label assigned to each topic, derived from the four most representative words identified through a class-based Term Frequency-Inverse Document Frequency analysis of the topic's content (Grootendorst, 2022)."),
            dcc.Markdown('''
            Note: The naming conventions for Topic and Topic_Label adhere to those established by BERTopic, utilizing capitalization for consistency and to facilitate future interactions with the library.
            ''')
        ]),
    ]),
    html.Hr(),

    # Libraries, Inspirations and Tools used
    html.H2("Built With", className="title"),
    html.Hr(),
    html.Ul([
        html.Li(html.A("Pre-processing with NLTK", href="https://github.com/nltk/nltk", target="_blank")),
        html.Li(html.A("Sentiment classification with VaderSentiment", href="https://github.com/cjhutto/vaderSentiment", target="_blank")),
        html.Li(html.A("Topic Modeling with BERTopic", href="https://github.com/MaartenGr/BERTopic", target="_blank")),
        html.Li(html.A("Dashboard Development with Dash (Plotly)", href="https://github.com/plotly/dash", target="_blank")),
        html.Li(html.A("Inspiration for ranger sliders from Dash Opioid Epidemic Demo", href="https://github.com/plotly/dash-opioid-epidemic-demo", target="_blank")),
        html.Li(html.A("Inspiration for tabs from Dash Manufacture SPC Dashboard", href="https://github.com/dkrizman/dash-manufacture-spc-dashboard", target="_blank")),
        html.Li("ChatGPT4 and Claude 3 Opus were utilised for code development and bug fixing.")
    ]),
    html.Hr(),

    # References
    html.H2("References", className="title"),
    html.Hr(),
    html.Ul([
        html.Li("Briggs, A. R., Clark, J., & Hall, I. (2012). Building bridges: understanding student transition to university. Quality in higher education, 18(1), 3-21."),
        html.Li("Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794."),
        html.Li("Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014."),
        html.Li("Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020, May). The pushshift reddit dataset. In Proceedings of the international AAAI conference on web and social media (Vol. 14, pp. 830-839)."),
        html.Li("Reddy, K. J., Menon, K. R., & Thattil, A. (2018). Academic stress and its sources among university students. Biomedical and pharmacology journal, 11(1), 531-537."),
        html.Li("Solatorio, A. V. (2024). GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning. arXiv preprint arXiv:2402.16829.")
    ]),
    html.Hr()
])



#################
# Topic Frequency Page
#################

topic_frequency_page = html.Div([
    # Page Title
    html.H1("Tracking Topic Frequencies over Time", className="title"),
    html.Hr(),
    html.P([
    "(1) Select Range of Topics.", 
    html.Br(),
    "(2) Select Type of Frequency: Absolute Counts or Normalized Percentages (% of frequency for the topic out of all selected topics)."
    ], className="instructions"),
    # Page Topic Slider
    dcc.RangeSlider(
        id='topic-range-slider',
        min=1,
        max=topic_max,
        value=[1, 10],
        marks={**{1: '1'}, **{i: str(i) for i in range(10, topic_max, 10)}, **{topic_max: str(topic_max)}},
        step=None
    ),
    # Page Frequency Tabs
    dcc.Tabs(
        id='frequency-tabs',
        value='absolute',
        children=[
            dcc.Tab(label='Absolute Counts', value='absolute'),
            dcc.Tab(label='Normalized Percentages', value='normalized')
        ]
    ),
    # Page Visualisation
    html.Div(
        className='graph-container',
        children=[dcc.Graph(id='topic-frequency-graph')]
    )
])



#################
# Sentiment Analysis Page
#################

sentiment_analysis_page = html.Div([
    # Page Title
    html.H1(id='sentiment-analysis-title', className="title"),
    html.Hr(),
    html.P([
    "(1) Select Topic of Interest.", 
    html.Br(),
    "(2) Select Type of Frequency: Absolute Counts or Normalized Percentages (% of frequency for the topic out of all selected topics)."
    ], className="instructions"),
    # Page Topic Dropdown List
    dcc.Dropdown(
        id='topic-dropdown',
        options=dropdown_options,
        value=topic_label_df['Topic_Label'].iloc[0],
        className='Select',
        clearable=False
    ),
    # Page Frequency Tabs
    dcc.Tabs(
        id='sentiment-frequency-tabs',
        value='absolute',
        children=[
            dcc.Tab(label='Absolute Counts', value='absolute'),
            dcc.Tab(label='Normalized Percentages', value='normalized')
        ]
    ),
    # Page Visualisation
    html.Div(
        className='graph-container',
        children=[dcc.Graph(id='sentiment-analysis-graph')]
    )
])



#################
# Topic Data Page
#################
topic_data_page = html.Div([
    # Page Title
    html.H1(id='topic-data-title', className="title"),
    html.Hr(),
    html.P([
    "(1) Select Topic of Interest.", 
    html.Br(),
    "(2) Select Range of Years."
    ], className="instructions"),
    # Page Topic Dropdown List
    dcc.Dropdown(
        id='topic-data-dropdown',
        options=dropdown_options,
        value=topic_label_df['Topic_Label'].iloc[0],
        style={'marginBottom': '20px'},
        className='Select',
        clearable=False
    ),
    # Page Year Slider
    dcc.RangeSlider(
        id='year-range-slider',
        min=2016,
        max=2023,
        value=[2016, 2023],
        marks={str(year): str(year) for year in range(2016, 2024)},
        step=1,
        className='year-slider'
    ),
    # Topic Table Legend
    html.Div([
        html.Div([
            html.P("Content Sentiment:", className="topic-table-legend-title"),
            html.Span("Positive", className="topic-table-legend positive"),
            html.Span("Neutral", className="topic-table-legend neutral"),
            html.Span("Negative", className="topic-table-legend negative")
        ], className="topic-table-legend")
    ]),

    # Topic Table
    html.Div(
        className='table-container',
        children=[html.Div(id='topic-data-table')]
    )
])



#################
# Interpretation Page
#################

interpretation_page = html.Div([
    html.H2("Interpretation, Limitations and Future Direction", className="title"),        
    html.Hr(),
    # Sample from Topic Frequency Visualisation
    html.Div([
        dcc.Graph(figure=generate_topic_frequency_html(sentiment_data, topic_max)),
        html.Hr(),
    # Explanation 1
        dcc.Markdown('''
    The topic frequency graph provides an overview of the most prevalent themes discussed by UK university students on Reddit r/UniUK. 
    Notably, while absolute count shows an increasing trend for all topics as the subreddit grows in popularity (see Topic Frequency page), normalizing the data allows us to identify the relative prominence of each topic over time. 
    Among the top topics, we find a mix of academic concerns (e.g., accommodation, university applications), social aspects (e.g., making friends, societies), and personal well-being (e.g., mental health, finance).
    First, the prominence of topics such as job, university choice and application, accommodation and finance highlights the practical challenges that students face in their university journey.
    Likely, students view such social media platforms as an avenue of seeking help and advice from other UK university students who have experienced similar concerns.
    These discussions not only provide valuable peer-to-peer support but also offer insights into the common struggles and decision-making processes of students.
    '''),
    dcc.Markdown('''
    Interestingly, discussions related to mental health, general health and disabilities (Topic 8) also consistently rank among the most frequently discussed issues. 
    This observation aligns with the growing concern of student mental health in higher education (Gagné et al., 2021). 
    To further understand how these data can help us comprehend the experiences and needs of UK university students, we will delve deeper into the mental health topic as a specific area of interest in the subsequent analysis.
    ''')
    ,]),
    html.Hr(),
    # Sample from Sentiment Analysis Visualisation
    html.Div([
        dcc.Graph(figure=generate_sentiment_analysis_html(sentiment_data)),
        html.Hr(),
    # Explanation 2
        dcc.Markdown('''
        Zooming in on the sentiment analysis graph for Topic 8 (mental health, general health, and disability), we observe an interesting trend. 
        While the absolute number of posts related to this topic is increasing (see Sentiment Analysis page), the proportion of positive, neutral, and negative sentiments remains relatively stable over time. 
        Notably, positive posts consistently outnumber negative ones by a ratio of nearly 2:1.
        This finding suggests that while mental health concerns are indeed prevalent among UK university students, as corroborated by existing research (Gagné et al., 2021), the discourse on Reddit appears to have a more supportive and encouraging tone. 
        The higher proportion of positive posts could indicate that students are finding solace, advice, and a sense of community through these online discussions.
    '''),
    ]),
    html.Hr(),
    # Sample from Topic Data Table
    html.Div([
        html.P("Topic Data - Topic 8: Mental, Health, Adhd, Gp", className="title-center"), 
        html.Div([
            generate_topic_data_table(sentiment_data)
        ], style={'height': '600px', 'overflow': 'auto'}),
        html.Hr(),
    # Explanation 3
        dcc.Markdown('''
        Looking at the data table, a closer examination of the content within Topic 8 reveals that students are increasingly turning to the r/UniUK forum to share their mental health struggles, seek advice, and offer support to their peers (page 3 of Topic 8 shown here; see Topic Data page for more). 
        The posts highlight the various stressors that students face, such as academic pressure, social isolation, and financial worries. 
        However, the responses to these posts often contain words of encouragement, practical tips, and reminders to prioritize self-care and seek professional help when needed.
        This content analysis reinforces the notion that online social media platforms like Reddit are emerging as important channels for students to connect, share their experiences, and find support outside of the traditional university mental health support systems. 
        As Winstone et al. (2021) note, youths are increasingly relying on social media for social connectedness in the face of well-being challenges.
        The analysis of social media discourse on r/UniUK thus provides insights into how these platforms can complement traditional mental health support systems in better understanding and supporting UK university students' well-being needs.
        Notably, this dashboard is an exploratory tool to examine various topics through topic frequencies and sentiment trends, offering a nuanced understanding of the authentic student experience. 
        Future research could extend this tool to investigate integrating social media insights into student policies and intervention plans to better support university life.
    ''')
    ]),
    html.Hr(),
    
    # Limitations and Future Direction
    html.P("Limitations and Future Direction",className="title"),
    html.Hr(),
    dcc.Markdown('''
    This study employed pre-trained transformer-based models with BerTopic for topic classification, which may have limitations in capturing complex nuances between topics (Souza & Filho, 2022). 
    For example, Topic 9 - Math, Physic, Level, Alevel - may actually belong to 2 separate categories of (1) type of subjects and (2) pre-university education.
    To enhance the accuracy of topic classification, future research could consider pre-training models with domain-specific data, such as trained twitter datasets relating to students, to better capture the intricacies of these social media data (Guo et al., 2022). 
    Similarly, while lexicon-based and rule-based techniques like VADER are useful for sentiment analysis without training data, they may struggle with sentences containing sarcasm or irony, which fall outside of the predefined rules (Al-Natour & Turetken, 2020; Samaras, García-Barriocanal & Sicilia, 2023). 
    To address this limitation, future studies could explore fine-tuning models using Reddit-specific training data to improve sentiment classification accuracy.
    '''),
    dcc.Markdown('''
    In addition to improving classification accuracy, topic representation could be enhanced using Large Language Models (LLMs) to re-examine the identified samples within each topic and generate more representative labels (Grootendorst, 2023). 
    For instance, Topic 2 in the current study, which classifies different cities or university names like City, University, Manchester, and Leeds, could be more accurately labeled as "City or University Choices" using LLMs. 
    This approach would provide a more comprehensive understanding of the overarching themes within each topic.
    Besides that, LLMs also opens up the possibilities in the exploration and interpretation of the data with the integration of AI-powered data visualizations. 
    Leveraging LLMs like the ChatGPT API, researchers could create open-ended, prompt-based visualizations that are generated on-the-fly based on user prompts (Biswas, 2023). 
    This approach would enable users to ask more targeted, exploratory questions, generate custom visualisations and uncover novel insights tailored to their specific interests.
    '''),     
    # Example of AI-powered Visualisation  
    html.Figure([
        html.Img(src='/assets/future_direction.gif', style={'width': '70%', 'height': '70%'}),
        html.Figcaption("Example of AI-powered visualisation from Biswas (2023).")
    ]),
    html.Hr(),
    
    # References
    html.P("References", className="title"),
    html.Hr(),    
    html.Li("Al-Natour, S., & Turetken, O. (2020). A comparative assessment of sentiment analysis and star ratings for consumer reviews. International Journal of Information Management, 54, 102132.", className="reference"),
    html.Li("Biswas, A. (2023, October 17). AI-powered data visualizations: Introducing an app to generate charts using only a single prompt and OpenAI large language models. Databutton. https://medium.com/databutton/ai-powered-data-visualization-134e89d82d99", className="reference"),
    html.Li("Gagné, T., Schoon, I., McMunn, A., & Sacker, A. (2021). Mental distress among young adults in Great Britain: long-term trends and early changes during the COVID-19 pandemic. Social Psychiatry and Psychiatric Epidemiology, 1-12.", className="reference"),
    html.Li("Grootendorst, M. (2023, August 22). Topic modeling with Llama 2: Create easily interpretable topics with Large Language Models. Towards Data Science. https://towardsdatascience.com/topic-modeling-with-llama-2-85177d01e174", className="reference"),
    html.Li("Guo, Y., Ge, Y., Yang, Y. C., Al-Garadi, M. A., & Sarker, A. (2022). Comparison of pretraining models and strategies for health-related social media text classification. In Healthcare (Vol. 10, No. 8, p. 1478). MDPI.", className="reference"),
    html.Li("Samaras, L., García-Barriocanal, E., & Sicilia, M. A. (2023). Sentiment analysis of COVID-19 cases in Greece using Twitter data. Expert Systems with Applications, 230, 120577.", className="reference"),
    html.Li("Souza, F. D., & Filho, J. B. D. O. E. S. (2022). BERT for sentiment analysis: pre-trained and fine-tuned alternatives. In International Conference on Computational Processing of the Portuguese Language (pp. 209-218). Cham: Springer International Publishing.", className="reference"),
    html.Li("Winstone, L., Mars, B., Haworth, C. M., & Kidger, J. (2021). Social media use and social connectedness among adolescents in the United Kingdom: a qualitative exploration of displacement and stimulation. BMC public health, 21, 1-15.", className="reference")
])



#################
# Callbacks
#################

# Callback to update page content based on page choice
@app.callback(Output('page-content', 'children'),
              [Input('app-tabs', 'value')])
def display_page(tab):
    if tab == 'tab0':
        return background_page
    elif tab == 'tab1':
        return topic_frequency_page
    elif tab == 'tab2':
        return sentiment_analysis_page
    elif tab == 'tab3':
        return topic_data_page
    elif tab == 'tab4':
        return interpretation_page
    else:
        return background_page

# Callback for topic frequency graph
@app.callback(
    Output('topic-frequency-graph', 'figure'),
    [Input('topic-range-slider', 'value'),
     Input('frequency-tabs', 'value')]
)



#################
# Topic Frequency Visualisation
#################

def update_topic_frequency_graph(selected_range, frequency_type):
    """
    Update the topic frequency graph based on the selected topic range and frequency type.

    Args:
        selected_range (list): A list containing the start and end values of the selected topic range.
        frequency_type (str): The type of frequency to display ('absolute' or 'normalized').

    Returns:
        go.Figure: The updated topic frequency graph figure.
    """

    # Filter data based on selected topic range
    filtered_data = sentiment_data[sentiment_data['Topic'].isin(range(selected_range[0], selected_range[1] + 1))]

    # Ensure the index is a DatetimeIndex before converting to period
    if not isinstance(filtered_data.index, pd.DatetimeIndex):
        filtered_data['created_utc'] = pd.to_datetime(filtered_data['created_utc'])
        filtered_data.set_index('created_utc', inplace=True)
        
    # Create a 'Quarter' column from the 'created_utc' index
    filtered_data['Quarter'] = filtered_data.index.to_period('Q')

    # Aggregate data by Quarter and Topic_Label
    topic_freq_over_time = filtered_data.groupby(['Quarter', 'Topic_Label']).size().unstack(fill_value=0)

    # Normalize frequencies by quarter and multiply by 100 for percentage
    topic_freq_over_time_normalized = topic_freq_over_time.div(topic_freq_over_time.sum(axis=1), axis=0) * 100

    # Function to extract topic number after removing "Topic " prefix
    def extract_topic_number(topic_label):
        try:
            return int(topic_label.split(":")[0].split(" ")[1])
        except (ValueError, IndexError):
            return float('inf')

    # Sort the columns (topics) based on the extracted topic number
    sorted_columns = sorted(topic_freq_over_time.columns, key=extract_topic_number)

    # Reorder DataFrame columns according to the sorted topics
    topic_freq_over_time = topic_freq_over_time[sorted_columns]
    topic_freq_over_time_normalized = topic_freq_over_time_normalized[sorted_columns]

    # Initialize figure
    fig = go.Figure()
    
    # Add traces for absolute and normalized frequencies, with sorting applied to columns
    for topic_label in sorted_columns:
        fig.add_trace(go.Scatter(
            x=topic_freq_over_time.index.to_timestamp(),
            y=topic_freq_over_time[topic_label],
            mode='lines+markers',
            name=topic_label,
            hoverinfo='x+y',
            hovertemplate=f"{topic_label}<br>Frequency: %{{y}}<br>Quarter: %{{x}}<extra></extra>",
            visible=frequency_type == 'absolute'
        ))

        fig.add_trace(go.Scatter(
            x=topic_freq_over_time_normalized.index.to_timestamp(),
            y=topic_freq_over_time_normalized[topic_label],
            mode='lines+markers',
            name=f"{topic_label} (Normalized)",
            hoverinfo='x+y',
            hovertemplate=f"{topic_label}<br>Frequency: %{{y:.1f}}%<br>Quarter: %{{x}}<extra></extra>",
            visible=frequency_type == 'normalized'
        ))
    
    # Additional inputs like axis legends
    fig.update_layout(
        xaxis_title="<b>Time</b>",
        yaxis_title="<b>Frequency</b>",
        legend_title="<b>Topic Label</b>",
        template="plotly_dark",
        margin=dict(t=30, b=55, l=0, r=0),
    )

    return fig

# Callback for sentiment analysis graph
@app.callback(
    [Output('sentiment-analysis-title', 'children'),
     Output('sentiment-analysis-graph', 'figure')],
    [Input('topic-dropdown', 'value'),
     Input('sentiment-frequency-tabs', 'value')]
)



#################
# Sentiment Analysis Visualisation
#################

def update_sentiment_analysis_graph(selected_topic_label, frequency_type):
    """
    Update the sentiment analysis graph based on the selected topic and frequency type.

    Args:
        selected_topic_label (str): The selected topic label from the dropdown.
        frequency_type (str): The type of frequency to display ('absolute' or 'normalized').

    Returns:
        tuple: A tuple containing the updated title and figure for the sentiment analysis graph.
    """

    # Absolute Frequencies
    # Reset the index 'created_utc' 
    if sentiment_data.index.name == 'created_utc':
        sentiment_data.reset_index(inplace=True)
    
    # Group 'created_utc', 'Topic_Label', and 'sentiment' for analysis
    sentiment_counts = sentiment_data.groupby(
        [pd.Grouper(key='created_utc', freq='Q'), 'Topic_Label', 'sentiment']
    ).size().unstack(fill_value=0).reset_index()
    
    # Extract topic numbers for filtering
    sentiment_counts['Topic_Number'] = sentiment_counts['Topic_Label'].apply(
        lambda x: int(x.split(':')[0].replace('Topic ', '').strip())
    )
    
    # Filter for the selected topic
    filtered_sentiment_counts = sentiment_counts[
        sentiment_counts['Topic_Label'] == selected_topic_label
    ].copy()
    
    # Get the actual Topic_Label for the selected topic (for title)
    topic_label = filtered_sentiment_counts['Topic_Label'].iloc[0]
    
    # Define colors for each sentiment
    colors = {'negative': '#FF0000', 'neutral': '#FFFF00', 'positive': '#00FF00'}

    # Plot for Absolute Frequencies
    fig_abs = go.Figure()
    
    # Add traces for each sentiment for the selected topic
    for sentiment in ['positive', 'neutral', 'negative']:
        fig_abs.add_trace(
            go.Scatter(
                x=filtered_sentiment_counts['created_utc'],
                y=filtered_sentiment_counts[sentiment],
                mode='lines+markers',
                name=sentiment,
                legendgroup=topic_label,
                line=dict(color=colors[sentiment]),
                visible=frequency_type == 'absolute'
            )
        )
    
    # Normalized Frequencies
    # Reset the DataFrame index 'created_utc' 
    if 'created_utc' not in filtered_sentiment_counts.columns:
        filtered_sentiment_counts.reset_index(inplace=True)
    
    # Check if 'total' column exists already, if not, calculate and merge it
    if 'total' not in filtered_sentiment_counts.columns:
        # Calculate the total sentiments by summing negative, neutral, and positive columns
        filtered_sentiment_counts.loc[:, 'total'] = filtered_sentiment_counts[['negative', 'neutral', 'positive']].sum(axis=1)
    
    # Normalize sentiments
    for sentiment in ['negative', 'neutral', 'positive']:
        normalized_column_name = f'{sentiment}_normalized'
        filtered_sentiment_counts.loc[:, normalized_column_name] = (filtered_sentiment_counts[sentiment] / filtered_sentiment_counts['total']) * 100
    
    fig_norm = go.Figure()
    
    # Add traces for each sentiment for the selected topic
    for sentiment in ['positive', 'neutral', 'negative']:
        normalized_column_name = f'{sentiment}_normalized'
        fig_norm.add_trace(
            go.Scatter(
                x=filtered_sentiment_counts['created_utc'],
                y=filtered_sentiment_counts[normalized_column_name],
                mode='lines+markers',
                name=f'{sentiment} (Normalized)',
                legendgroup=topic_label,
                line=dict(color=colors[sentiment]),
                visible=frequency_type == 'normalized'
            )
        )
    
    # Initialize the final figure
    fig = go.Figure()

    # Add traces from both absolute and normalized figures
    fig.add_traces(fig_abs.data + fig_norm.data)
    
    # Additional inputs like axis legends
    fig.update_layout(
        xaxis_title='<b>Time</b>',
        yaxis_title='<b>Frequency</b>',
        legend_title='<b>Sentiment</b>',
        legend=dict(y=0.5,font=dict(size=12)),
        template="plotly_dark",
        margin=dict(t=30, b=55, l=0, r=0)
    )
    
    # 1st output for Title, 2nd for Figure
    return f"Tracking Sentiment over Time for {topic_label}", fig

# Callback for topic table
@app.callback(
    [Output('topic-data-title', 'children'),
     Output('topic-data-table', 'children')],
    [Input('topic-data-dropdown', 'value'),
     Input('year-range-slider', 'value')]
)



#################
# Topic Table Visualisation
#################

def update_topic_data(selected_topic_label, year_range):
    """
    Update the topic data table based on the selected topic label and year range.

    Args:
        selected_topic_label (str): The selected topic label from the dropdown.
        year_range (list): A list containing the start and end years of the selected range.

    Returns:
        tuple: A tuple containing the updated title and table for the topic data.
    """

    # Check if sentiment_data is empty
    if sentiment_data.empty:
        return "No data available", None

    # Check if the selected topic label exists in the DataFrame
    if selected_topic_label not in sentiment_data['Topic_Label'].values:
        return f"Topic {selected_topic_label} not found", None

    # Get the Topic_Label for the selected topic
    topic_label = selected_topic_label

    # Filter sentiment_data based on the selected topic label
    filtered_data = sentiment_data[sentiment_data['Topic_Label'] == selected_topic_label]

    # Reset the index to make 'created_utc' a regular column for display
    filtered_data = filtered_data.reset_index()

    # Format 'created_utc' as 'MMM YYYY'
    filtered_data['created_utc'] = filtered_data['created_utc'].dt.strftime('%b %Y')

    # Extract year for filtering
    filtered_data['Year'] = filtered_data['created_utc'].apply(lambda x: int(x.split(' ')[1]))

    # Filter based on selected year range
    filtered_data = filtered_data[(filtered_data['Year'] >= year_range[0]) & (filtered_data['Year'] <= year_range[1])]

    # Rename columns for the table display
    filtered_data = filtered_data.rename(columns={'created_utc': 'Date', 'body': 'Content'})

    # Apply conditional styling based on sentiment
    styles = [
        {
            'if': {'filter_query': '{sentiment} = "positive"'},
            'backgroundColor': '#B0DBA4',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        },
        {
            'if': {'filter_query': '{sentiment} = "negative"'},
            'backgroundColor': '#FF887E',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        },
        {
            'if': {'filter_query': '{sentiment} = "neutral"'},
            'backgroundColor': '#FEE191',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        }
    ]

    # Display content
    desired_columns = ['Date', 'Content', 'sentiment']
    table = html.Div(
        className='table-container',
        children=[
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in desired_columns],
                data=filtered_data.to_dict('records'),
                page_size=10,
                style_data_conditional=styles,
                style_cell_conditional=[
                    {'if': {'column_id': 'Date'}, 'width': '7%', 'fontSize': '16px', 'textAlign': 'left'},
                    {'if': {'column_id': 'Content'}, 'whiteSpace': 'normal', 'textOverflow': 'ellipsis', 'width': '92.9%', 'fontSize': '16px', 'textAlign': 'left'},
                    {'if': {'column_id': 'sentiment'}, 'width': '0.1%', 'textAlign': 'left'}
                ]
            )
        ]
    )

    return f"Topic Data - {topic_label}", table



# Running the app
if __name__ == '__main__':
    app.run_server(debug=False)
