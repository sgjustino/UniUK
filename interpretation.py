import plotly.graph_objs as go
import pandas as pd
from dash import dash_table
from dash import html

def generate_topic_frequency_html(sentiment_data, topic_max):
    """
    Generate the topic frequency figure in HTML format.

    Args:
        sentiment_data (pd.DataFrame): The sentiment data DataFrame.
        topic_max (int): The maximum number of topics to consider.

    Returns:
        go.Figure: The topic frequency figure.
    """
    
    # Generate the topic frequency figure
    fig = go.Figure()
    
    # Add traces for normalized frequencies with all topics
    filtered_data = sentiment_data[sentiment_data['Topic'].isin(range(1, topic_max))]
    
    # Ensure the index is a DatetimeIndex before converting to period
    if not isinstance(filtered_data.index, pd.DatetimeIndex):
        filtered_data['created_utc'] = pd.to_datetime(filtered_data['created_utc'])
        filtered_data.set_index('created_utc', inplace=True)
    
    # Create a 'Quarter' column from the 'created_utc' index
    filtered_data['Quarter'] = filtered_data.index.to_period('Q').astype(str)
    
    topic_freq_over_time = filtered_data.groupby(['Quarter', 'Topic_Label']).size().unstack(fill_value=0)
    topic_freq_over_time_normalized = topic_freq_over_time.div(topic_freq_over_time.sum(axis=1), axis=0) * 100
    
    # Sort the columns (topics) based on the extracted topic number
    sorted_columns = sorted(topic_freq_over_time.columns, key=lambda x: int(x.split(':')[0].split(' ')[1]))
    
    # Reorder DataFrame columns according to the sorted topics
    topic_freq_over_time = topic_freq_over_time[sorted_columns]
    topic_freq_over_time_normalized = topic_freq_over_time_normalized[sorted_columns]
    
    for topic_label in topic_freq_over_time_normalized.columns:
        fig.add_trace(go.Scatter(
            x=topic_freq_over_time_normalized.index,
            y=topic_freq_over_time_normalized[topic_label],
            mode='lines+markers',
            name=topic_label
        ))
    
    # Additional layout and styling
    fig.update_layout(
        title=dict(
            text="Tracking Topic Frequencies over Time",
            x=0.5,
            font=dict(size=24)
        ),
        xaxis_title="<b>Time</b>",
        yaxis_title="<b>Normalized Frequency</b>",
        legend_title="<b>Topic Label</b>",
        template="plotly_dark",
        margin=dict(t=80, b=80, l=0, r=0),
    )
    
    # Save the figure as an HTML file
    fig.write_html("fig/topic_freq_example.html")
    
    return fig

def generate_sentiment_analysis_html(sentiment_data):
    """
    Generate the sentiment analysis figure in HTML format.

    Args:
        sentiment_data (pd.DataFrame): The sentiment data DataFrame.

    Returns:
        go.Figure: The sentiment analysis figure.
    """

    # Generate the sentiment analysis figure
    fig = go.Figure()
    
    # Add traces for normalized frequencies for "Topic 8: Mental, Health, Adhd, Gp"
    filtered_sentiment_counts = sentiment_data[sentiment_data['Topic_Label'] == "Topic 8: Mental, Health, Adhd, Gp"].copy()
    
    # Ensure the index is a DatetimeIndex before converting to period
    if not isinstance(filtered_sentiment_counts.index, pd.DatetimeIndex):
        filtered_sentiment_counts['created_utc'] = pd.to_datetime(filtered_sentiment_counts['created_utc'])
        filtered_sentiment_counts.set_index('created_utc', inplace=True)
    
    # Create a 'Quarter' column from the 'created_utc' index
    filtered_sentiment_counts['Quarter'] = filtered_sentiment_counts.index.to_period('Q').astype(str)
    
    # Group by 'Quarter' and 'sentiment' and count the occurrences
    sentiment_counts = filtered_sentiment_counts.groupby(['Quarter', 'sentiment']).size().unstack(fill_value=0)
    sentiment_counts = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    
    colors = {'negative': '#FF0000', 'neutral': '#FFFF00', 'positive': '#00FF00'}

    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in sentiment_counts.columns:
            fig.add_trace(go.Scatter(                
                x=sentiment_counts.index,
                y=sentiment_counts[sentiment],
                mode='lines+markers',
                name=sentiment,
                marker=dict(color=colors[sentiment])
            ))
    
    # Additional layout and styling
    fig.update_layout(
        title=dict(
            text="Tracking Sentiment over Time for Topic 8: Mental, Health, Adhd, Gp",
            x=0.5,
            font=dict(size=24)
        ),
        xaxis_title='<b>Time</b>',
        yaxis_title='<b>Normalized Frequency</b>',
        legend_title='<b>Sentiment</b>',
        template="plotly_dark",
        margin=dict(t=80, b=80, l=0, r=0)
    )
    
    # Save the figure as an HTML file
    fig.write_html("fig/sentiment_analysis_example.html")
    
    return fig

def generate_topic_data_table(sentiment_data):
    """
    Generate the topic data table in HTML format.

    Args:
        sentiment_data (pd.DataFrame): The sentiment data DataFrame.

    Returns:
        dash_table.DataTable: The topic data table.
    """
    
    # Filter sentiment_data for Topic 8
    filtered_data = sentiment_data[sentiment_data['Topic_Label'] == "Topic 8: Mental, Health, Adhd, Gp"]

    # Reset the index to make 'created_utc' a regular column for display
    filtered_data = filtered_data.reset_index()

    # Format 'created_utc' as 'MMM YYYY'
    filtered_data['created_utc'] = filtered_data['created_utc'].dt.strftime('%b %Y')

    # Extract year for filtering
    filtered_data['Year'] = filtered_data['created_utc'].apply(lambda x: int(x.split(' ')[1]))

    # Filter data for page 5 (assuming 10 rows per page)
    start_index = 40
    end_index = 50
    filtered_data = filtered_data.iloc[start_index:end_index]

    # Rename columns for the table display
    filtered_data = filtered_data.rename(columns={'created_utc': 'Date', 'body': 'Content', 'sentiment': 'Sentiment'})

    # Define the styles for data conditional formatting
    styles = [
        {
            'if': {'filter_query': '{Sentiment} = "positive"'},
            'backgroundColor': '#B0DBA4',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        },
        {
            'if': {'filter_query': '{Sentiment} = "negative"'},
            'backgroundColor': '#FF887E',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        },
        {
            'if': {'filter_query': '{Sentiment} = "neutral"'},
            'backgroundColor': '#FEE191',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        }
    ]

    # Create the table
    desired_columns = ['Date', 'Content', 'Sentiment']
    table = html.Div(
        className='table-container',
        children=[
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in desired_columns],
                data=filtered_data.to_dict('records'),
                style_header={
                    'backgroundColor': 'black',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'left',
                    'fontFamily': 'Lato, sans-serif'
                },
                style_data_conditional=styles,
                style_cell_conditional=[
                    {'if': {'column_id': 'Date'}, 'width': '7%', 'fontSize': '16px', 'textAlign': 'left'},
                    {'if': {'column_id': 'Content'}, 'whiteSpace': 'normal', 'textOverflow': 'ellipsis', 'width': '92.9%', 'fontSize': '16px', 'textAlign': 'left'},
                    {'if': {'column_id': 'sentiment'}, 'width': '0.1%', 'textAlign': 'left'}
                ]
            )
        ]
    )

    # Create a figure to display the table
    fig = go.Figure(data=[go.Table(
        columnwidth=[70, 929, 1],
        header=dict(
            values=desired_columns,
            fill_color=['black'] * len(desired_columns),
            font=dict(color='white', size=12),
            align=['left'] * len(desired_columns)
        ),
        cells=dict(
            values=[filtered_data[col] for col in desired_columns],
            fill_color=[filtered_data['Sentiment'].map({'positive': '#B0DBA4', 'negative': '#FF887E', 'neutral': '#FEE191'})],
            font=dict(color='black', size=11),
            align=['left'] * len(desired_columns)
        )
    )])

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=0, b=0, l=0, r=0),
    )

    # Save the figure as an HTML file
    fig.write_html("fig/topic_data_example.html")

    return table