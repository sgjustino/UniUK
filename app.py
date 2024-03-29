# Dependencies
import pandas as pd 
import os
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html as html
from dash.dependencies import Input, Output
from dash import dash_table
from interpretation import generate_topic_frequency_gif, generate_sentiment_analysis_gif, generate_topic_data_table
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#################
# Data Processing
#################

'''
Load Data, remove error data, convert date correctly, merge Topic 1 to outliers and shift topic numbers sequentially
'''

# Build the path to the file
file_path = os.path.join(os.getcwd(), 'data', 'uniuk_sentiment_data.csv')

# Read the CSV file into a DataFrame
sentiment_data = pd.read_csv(file_path)

# Remove error data
sentiment_data = sentiment_data[sentiment_data['Topic_Label'].notna()]

#convert the 'created_utc' column to datetime format
sentiment_data['created_utc'] = pd.to_datetime(sentiment_data['created_utc'])

# Update Topic and Topic_Label for rows with Topic 99 to Topic 75
# Shift Topic 99 down to prevent range break
# Merge Topic 1 to outliers as Topic 1 (Thank, Thanks, Comment, Post) is not useful
sentiment_data.loc[sentiment_data['Topic'] == 99, 'Topic'] = 75
sentiment_data.loc[sentiment_data['Topic'] == 1, 'Topic'] = 75
sentiment_data.loc[sentiment_data['Topic'] == 75, 'Topic_Label'] = 'Topic 75: Outliers'

# Decrement all Topic values by 1
sentiment_data['Topic'] -= 1

# Update Topic_Label for all rows to reflect the decremented Topic
sentiment_data['Topic_Label'] = sentiment_data['Topic_Label'].apply(
    lambda x: f"Topic {int(x.split(':')[0].split(' ')[1]) - 1}:{x.split(':')[1]}"
)

#################
# Dashboard App 
#################

app = dash.Dash(__name__, external_stylesheets=["assets/styles.css"], suppress_callback_exceptions=True)
server = app.server
pd.options.mode.chained_assignment = None

# Set topic range
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
                    html.H2('R/UniUK Sentiment Dashboard'),
                    html.P(
                        'Examining key themes and sentiment trends from r/UniUK posts (2016-2023) to understand the changing perspectives and emotional dynamics of UK university students.',
                        style={'fontSize': '14px', 'color': '#aaaaaa'}
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
                    value="tab1",
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
    html.H2("Research Question: What are university students in the UK thinking and how do their sentiments change over time?", className="title"),    
    html.H2("Data Source", className="title"),
    dcc.Markdown('''
    The data prior to April 2023 was collected by an open-source project called Pushshift and hosted on academictorrents.com.
    https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10
    '''),

    html.H2("Data Preprocessing", className="title"),
    dcc.Markdown('''
    The data preprocessing steps, including text refinement using the NLTK library, sentiment analysis using VaderSentiment, and topic modeling using BerTopic, are not covered in this repository. 
    The preprocessing code can be found at https://www.kaggle.com/code/sgjustino/uniuk-topic-modeling-with-bertopic?scriptVersionId=168342984.
    '''),

    html.H2("Meta-Data for Dashboard", className="title"),
    html.Div([
        html.P("Here is a preview of the data with an explanation of each column:"),
        dash_table.DataTable(
            data=sentiment_data.head().to_dict('records'),
            columns=[{"name": i, "id": i} for i in sentiment_data.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': 'black',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'fontFamily': 'Lato, sans-serif',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            }
        ),
        html.Ul([
            html.Li("created_utc: The timestamp of when the post was created."),
            html.Li("body: The content of the post."),
            html.Li("sentiment: The sentiment of the post as determined by VaderSentiment (positive, neutral, or negative)."),
            html.Li("Topic: The topic number that the post belongs to as determined by BerTopic."),
            html.Li("Topic_Label: The descriptive label for the topic.")
        ])
    ]),

    html.H2("Dashboard Explanation", className="title"),
    dcc.Markdown('''
    The visualization part of the project showcases the identified themes and their popularity over time, as well as the sentiment (positive, neutral, negative) associated with each theme. It also includes a data table to explore the actual data. The dashboard is built using Dash (Plotly).
    '''),

    html.H2("Learn More", className="title"),
    html.P(html.A("Click here", href="https://github.com/sgjustino/UniUK", target="_blank"), " to find out more information about this project."),

    html.H2("Built With", className="title"),
    html.Ul([
        html.Li(html.A("NLTK", href="https://github.com/nltk/nltk", target="_blank")),
        html.Li(html.A("BERTopic", href="https://github.com/MaartenGr/BERTopic", target="_blank")),
        html.Li(html.A("VaderSentiment", href="https://github.com/cjhutto/vaderSentiment", target="_blank")),
        html.Li(html.A("Dash (Plotly)", href="https://github.com/plotly/dash", target="_blank")),
        html.Li(html.A("Dash Opioid Epidemic Demo (inspiration for sliders)", href="https://github.com/plotly/dash-opioid-epidemic-demo", target="_blank")),
        html.Li(html.A("Dash Manufacture SPC Dashboard (inspiration for tabs)", href="https://github.com/dkrizman/dash-manufacture-spc-dashboard", target="_blank"))
    ]),

    html.H2("References", className="title"),
    html.Ul([
        html.Li("Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014."),
        html.Li("Solatorio, A. V. (2024). GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning. arXiv preprint arXiv:2402.16829. https://arxiv.org/abs/2402.16829"),
        html.Li("Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.")
    ])
])

#################
# Topic Frequency Page
#################

index_page = html.Div([
    # Page Title
    html.H1("Tracking Topic Frequencies over Time", className="title"),
    html.P([
    "(1) Select Range of Topics.", 
    html.Br(),
    "(2) Select Type of Frequency: Absolute or Normalized (% frequency across selected topics)."
    ], style={
        'color': '#aaaaaa',  
        'fontWeight': 'bold',  
        'fontSize': '14px',
        'marginTop': '-5px' 
    }),
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
            dcc.Tab(label='Absolute Frequency', value='absolute'),
            dcc.Tab(label='Normalized Frequency', value='normalized')
        ]
    ),
    # Page Visualisation
    html.Div(
        className='graph-container',
        children=[dcc.Graph(id='topic-frequency-graph')]
    )
])

'''
Process Topics for dropdown list usage
'''

# Create a DataFrame with unique Topic_Label and their corresponding Topic values
topic_label_df = sentiment_data[['Topic', 'Topic_Label']].drop_duplicates()

# Sort the DataFrame by Topic values
topic_label_df = topic_label_df.sort_values('Topic')

# Create the options for the dropdown
dropdown_options = [{'label': row['Topic_Label'], 'value': row['Topic_Label']} for _, row in topic_label_df.iterrows()]

#################
# Sentiment Analysis Page
#################

sentiment_analysis_page = html.Div([
    # Page Title
    html.H1(id='sentiment-analysis-title', className="title"),
    html.P([
    "(1) Select Topic of Interest.", 
    html.Br(),
    "(2) Select Type of Frequency: Absolute or Normalized (% frequency across selected topics)."
    ], style={
        'color': '#aaaaaa',  
        'fontWeight': 'bold',  
        'fontSize': '14px',
        'marginTop': '-5px'
    }),
    # Page Topic Dropdown List
    dcc.Dropdown(
        id='topic-dropdown',
        options=dropdown_options,
        value=topic_label_df['Topic_Label'].iloc[0]
    ),
    # Page Frequency Tabs
    dcc.Tabs(
        id='sentiment-frequency-tabs',
        value='absolute',
        children=[
            dcc.Tab(label='Absolute Frequency', value='absolute'),
            dcc.Tab(label='Normalized Frequency', value='normalized')
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
    html.P([
    "(1) Select Topic of Interest.", 
    html.Br(),
    "(2) Select Range of Years."
    ],  style={
        'color': '#aaaaaa',  
        'fontWeight': 'bold',  
        'fontSize': '14px',
        'marginTop': '-5px' 
    }),
    # Page Topic Dropdown List
    dcc.Dropdown(
        id='topic-data-dropdown',
        options=dropdown_options,
        value=topic_label_df['Topic_Label'].iloc[0],
        style={'marginBottom': '20px'}
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
        html.P("Content Sentiment:", style={'font-weight': 'bold', 'margin-right': '10px'}),
        html.Span("Positive", style={'color': 'black', 'background-color': '#B0DBA4', 'padding': '2px 5px', 'margin-right': '10px'}),
        html.Span("Neutral", style={'color': 'black', 'background-color': '#FEE191', 'padding': '2px 5px', 'margin-right': '10px'}),
        html.Span("Negative", style={'color': 'black', 'background-color': '#FF887E', 'padding': '2px 5px'})
    ], style={'display': 'flex', 'align-items': 'center', 'margin-top': '0px'})
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
    html.H2("Research Question: What are university students in the UK thinking and how do their sentiments change over time?", className="title"),    
    html.P("First, we will talk about how the 3 Analysis Tabs (Topic Frequency, Sentiment Analysis, Topic Data) will allow a thorough examination of the Research Question."),
    
    html.Div([
        dcc.Graph(figure=generate_topic_frequency_gif(sentiment_data, topic_max)),
        html.P("Users can first look at the overall trends of topics that university students in UK are talking about on Reddit. However, as the popularity of the subreddit forum grows, absolute frequency would just show an increasing trend for all topics. Viewing the data by normalized frequency will allow us to see how topics trend relatively to other topics over time, to gauge or gain insights on the 'hottest' topics across time. Notably, the % is based on selected topics, and by selecting all 74 topics (including Topic 74 - Outliers), will allow us to examine the overall proportions of topics across all reddit activities on r/UniUK.")
    ], className="interpretation-section"),
    
    html.Div([
        dcc.Graph(figure=generate_sentiment_analysis_gif(sentiment_data)),
        html.P("Users can then zoom in on specific Topic of Interest to understand how sentiments within a topic trend over time. For example, looking at mental health/general health/disability as a topic (Topic 8), we can analyze how positive, neutral and negative posts trend across time. Notably, we can see that such posts are on an increasing uptrend, similar to other topics across the subreddit forum. It is easy to paint a congruent image when we compare with other research that shows that e.g. mental distress has been an increasing uptrend, especially with Covid-19 pandemic (Gagné et al., 2021). However, looking at the normalized frequency, we can see clearly that the proportion of sentiments has been relatively stable across time, with almost twice the number of positive posts to each negative post. Importantly, this does not refute the claims from research. But it does suggest that there is something different for online social media discourse relating to mental health."),
        html.P("Gagné, T., Schoon, I., McMunn, A., & Sacker, A. (2021). Mental distress among young adults in Great Britain: long-term trends and early changes during the COVID-19 pandemic. Social Psychiatry and Psychiatric Epidemiology, 1-12.", className="reference")
    ], className="interpretation-section"),
    
    html.Div([
        html.Div([
            generate_topic_data_table(sentiment_data)
        ], className="topic-data-table"),
        html.Div([
            html.P("Subsequently, such observations can be confirmed or further explored by diving into specific posts from Topic Data tab. The analysis here provide an interesting observation - students are increasingly posting about mental health topics (e.g. highlighting stress in school) and then receiving social support on an online forum (subreddit r/UniUK) and perhaps building a community that help each other, provide assurance and advice in dealing with mental health issues - with twice the number of positive posts to each negative post. While research showed that mental health concerns are on a rise, online social media discourse suggest an interesting channel of social support that are outside of current structured university mental health support system. As youths are increasingly using social media for social connectedness in countering mental health (Winstone et al., 2021), the brief analysis here suggest a growing necessity to encompass or even integrate social media platforms to synergize official university mental health support efforts."),
            html.P("Winstone, L., Mars, B., Haworth, C. M., & Kidger, J. (2021). Social media use and social connectedness among adolescents in the United Kingdom: a qualitative exploration of displacement and stimulation. BMC public health, 21, 1-15.", className="reference")
        ], className="topic-data-text")
    ], className="interpretation-section"),
    
    html.P("As stated, this project seeks to understand what university students in UK are thinking and how do their sentiments change over time. Using mental health as a topic, we provided an example of how social media discourse could be harnessed to understand an increasingly growing part of university students' lives - social media consumption. While this dashboard serves as an exploratory visualization to understand the research question, clearly from the mental health example, there is a need for more effort to understand social media discourse to better understand our university students."),
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
    elif tab == 'tab2':
        return sentiment_analysis_page
    elif tab == 'tab3':
        return topic_data_page
    elif tab == 'tab4':
        return interpretation_page
    else:
        return index_page

# Callback for topic frequency graph
@app.callback(
    Output('topic-frequency-graph', 'figure'),
    [Input('topic-range-slider', 'value'),
     Input('frequency-tabs', 'value')]
)

#################
# Topic Frequency Visualisation
#################

def update_figure(selected_range, frequency_type):
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
        legend=dict(
            font=dict(size=12),
            itemsizing='constant',
            itemwidth=30,
            traceorder='normal',
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='left',
            x=0
        ),
        template="plotly_dark",
        margin=dict(t=2, b=5, l=0, r=0),
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
    # Reset the index 'created_utc' 
    if sentiment_data.index.name == 'created_utc':
        sentiment_data.reset_index(inplace=True)
    
    # Group 'created_utc', 'Topic_Label', and 'sentiment' for Data Table
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
    
    # Get the actual Topic_Label for the selected topic
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
        margin=dict(t=2, b=5, l=0, r=0)
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
                style_header={
                    'backgroundColor': 'black',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'text-align': 'left',
                    'fontFamily': 'Lato, sans-serif'
                },

                # To Hide Sentiment Tab
                style_data_conditional=styles,
                style_cell_conditional=[
                    {'if': {'column_id': 'Date'},
                     'width': '7%',
                     'fontSize': '16px'},
                    {'if': {'column_id': 'Content'},
                     'whiteSpace': 'normal',
                     'textOverflow': 'ellipsis',
                     'fontSize': '16px'},
                    {'if': {'column_id': 'sentiment'},
                     'width': '0.1%'}
                ]
            )
        ]
    )

    return f"Topic Data - {topic_label}", table

if __name__ == '__main__':
    app.run_server(debug=False)