import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import pandas as pd
from dash import dash_table
from dash import html

def generate_topic_frequency_html(sentiment_data):
    """
    Generate the topic frequency figure in HTML format based on a fixed topic range, frequency type, and selected years.

    Args:
        sentiment_data (pd.DataFrame): The sentiment data DataFrame.

    Returns:
        go.Figure: The topic frequency figure.
    """
    selected_range = [1, 20]
    frequency_type = 'normalized'
    selected_years = [2016, 2023]

    # Filter data based on selected topic range and years
    filtered_data_years = sentiment_data[
        (sentiment_data['Topic'].isin(range(selected_range[0], selected_range[1] + 1))) &
        (sentiment_data['created_utc'].dt.year >= selected_years[0]) &
        (sentiment_data['created_utc'].dt.year <= selected_years[1])
    ]

    # Group by quarter and Topic_Label
    topic_freq_over_time = filtered_data_years.groupby(
        [pd.Grouper(key='created_utc', freq='Q'), 'Topic_Label']
    ).size().unstack(fill_value=0).reset_index()

    # Format the date
    topic_freq_over_time['created_utc'] = topic_freq_over_time['created_utc'].dt.strftime('%b %Y')

    # Function to extract topic number after removing "Topic " prefix
    def extract_topic_number(topic_label):
        try:
            return int(topic_label.split(":")[0].split(" ")[1])
        except (ValueError, IndexError):
            return float('inf')

    # Sort the columns (topics) based on the extracted topic number (ascending order)
    sorted_columns = sorted(topic_freq_over_time.columns[1:], key=extract_topic_number)

    # Initialize figure
    fig = go.Figure()
    
    def wrap_legend_label(label, max_width=20):
        words = label.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        lines.append(' '.join(current_line))
        return '<br>'.join(lines)

    for i, topic_label in enumerate(sorted_columns):
        wrapped_label = wrap_legend_label(topic_label)
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=topic_freq_over_time['created_utc'],
            y=topic_freq_over_time[topic_label],
            mode='lines',
            name=wrapped_label,
            stackgroup='one',
            groupnorm='percent' if frequency_type == 'normalized' else None,
            line=dict(width=1, color=color),
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.5)'),
            hoverinfo='name+x+y',
            hovertemplate='%{fullData.name}<br>Frequency: %{y:.0f}<br>Quarter: %{x}<extra></extra>',
        ))

    # Set y-axis title based on frequency type
    yaxis_title = "<b>Frequency</b>" if frequency_type == 'absolute' else "<b>Frequency (%)</b>"

    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(t=10),
        xaxis=dict(
            title="<b>Time</b>",
            automargin=True
        ),
        yaxis=dict(
            title=yaxis_title
        ),
        legend=dict(
            font=dict(size=10),
            traceorder="normal",
            itemwidth=30,
            itemsizing="constant",
        ),
        template="plotly_dark",
        hovermode="closest"
    )

    # Save the figure as an HTML file
    fig.write_html("fig/topic_freq_example.html")

    return fig


def generate_sentiment_analysis_html(sentiment_data):
    """
    Generate the sentiment analysis figure in HTML format based on a fixed topic, frequency type, and selected years.

    Args:
        sentiment_data (pd.DataFrame): The sentiment data DataFrame.

    Returns:
        go.Figure: The sentiment analysis figure.
    """
    selected_topic_label = "Topic 8: Mental, Health, Adhd, Gp"
    frequency_type = 'normalized'
    selected_years = [2016, 2023]

    # Reset the index 'created_utc' 
    if sentiment_data.index.name == 'created_utc':
        sentiment_data.reset_index(inplace=True)
    
    # Group 'created_utc', 'Topic_Label', and 'sentiment' for analysis
    sentiment_counts = sentiment_data.groupby(
        [pd.Grouper(key='created_utc', freq='Q'), 'Topic_Label', 'sentiment']
    ).size().unstack(fill_value=0).reset_index()
    
    # Format the date to show quarters
    sentiment_counts['created_utc'] = sentiment_counts['created_utc'].dt.strftime('%b %Y')
    
    # Filter for the selected topic and years
    filtered_sentiment_counts = sentiment_counts[
        (sentiment_counts['Topic_Label'] == selected_topic_label) &
        (sentiment_counts['created_utc'].str[-4:].astype(int) >= selected_years[0]) &
        (sentiment_counts['created_utc'].str[-4:].astype(int) <= selected_years[1])
    ].copy()
    
    # Get the actual Topic_Label for the selected topic (for title)
    topic_label = filtered_sentiment_counts['Topic_Label'].iloc[0]
    
    # Define colors for each sentiment (using the original colors)
    colors = {'negative': '#FF887E', 'neutral': '#FEE191', 'positive': '#B0DBA4'}

    # Function to convert hex to rgba
    def hex_to_rgba(hex_color, alpha=0.5):
        rgb = hex_to_rgb(hex_color)
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Initialize figure
    fig = go.Figure()
    
    # Add traces for each sentiment in the order: negative, neutral, positive
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment in filtered_sentiment_counts.columns:
            fig.add_trace(go.Scatter(
                x=filtered_sentiment_counts['created_utc'],
                y=filtered_sentiment_counts[sentiment],
                mode='lines',
                name=sentiment.capitalize(),
                stackgroup='one',
                groupnorm='percent' if frequency_type == 'normalized' else None,
                fillcolor=hex_to_rgba(colors[sentiment], 0.5),
                line=dict(color=colors[sentiment], width=2),
                hoverinfo='name+x+y',
                hovertemplate='%{fullData.name}<br>Frequency: %{y:.0f}<br>Quarter: %{x}<extra></extra>',
            ))
    
    # Set y-axis title based on frequency type
    yaxis_title = "<b>Frequency</b>" if frequency_type == 'absolute' else "<b>Frequency (%)</b>"
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(t=30, b=55, l=0, r=0),
        xaxis=dict(
            title="<b>Time (Quarters)</b>",
            tickmode='array',
            tickvals=filtered_sentiment_counts['created_utc'],
            ticktext=filtered_sentiment_counts['created_utc'],
            tickangle=45,
            automargin=True
        ),
        yaxis=dict(
            title=yaxis_title
        ),
        legend=dict(
            font=dict(size=10),
            traceorder="normal",
            itemwidth=30,
            itemsizing="constant",
        ),
        template="plotly_dark",
        hovermode="closest"
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