# Dependencies
import pandas as pd 
import os
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html as html
from dash.dependencies import Input, Output
from dash import dash_table
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
            html.H2('R/UniUK Sentiment Dashboard'),
            html.P(
                'Examining key themes and sentiment trends from r/UniUK posts (2016-2023) to understand the changing perspectives and emotional dynamics of UK university students.',
                style={'fontSize': '14px','marginTop': '-20px','color': '#aaaaaa', }
            ),
        # Learn More Button
        html.Div(
            style={'position': 'absolute', 'top': '20px', 'right': '10px'},
            children=[
                html.A(
                    html.Button("Learn More", 
                                className="button-primary", 
                                style={'fontSize': '16px', 'padding': '12px 18px'}),
                    href="https://github.com/sgjustino/UniUK"
                )
            ]
        ),
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
                    ],
                )
            ]
        ),
    ]),
    # Content Row (individual pages)
    html.Div(className='row', children=[
        html.Div(
            className='twelve columns',
            children=[html.Div(id='page-content')]
        )
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
        'marginTop': '-10px' 
    }),
    # Page Topic Slider
    dcc.RangeSlider(
        id='topic-range-slider',
        min=1,
        max=topic_max,  
        value=[0, 10],  
        marks={str(i): str(i) for i in range(0, topic_max + 1)},  
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
    dcc.Graph(id='topic-frequency-graph')
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
        'marginTop': '-10px'
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
    dcc.Graph(id='sentiment-analysis-graph')
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
        'marginTop': '-10px' 
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
    html.Div(id='topic-data-table')
])

#################
# Callbacks
#################

# Callback to update page content based on page choice
@app.callback(Output('page-content', 'children'),
              [Input('app-tabs', 'value')])
def display_page(tab):
    if tab == 'tab2':
        return sentiment_analysis_page
    elif tab == 'tab3':
        return topic_data_page
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
        legend=dict(y=0.5),
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
    for sentiment in ['negative', 'neutral', 'positive']:
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
    for sentiment in ['negative', 'neutral', 'positive']:
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
        legend=dict(y=0.5),
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
    table = dash_table.DataTable(
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
        # to hide sentiment tab
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

    # 1st output for Title, 2nd for table
    return f"Topic Data - {topic_label}", table

if __name__ == '__main__':
    app.run_server(debug=False) 