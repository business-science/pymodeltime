import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg

def plot_modeltime_forecast(df, date_col='date', value_col='value',
                            title="Model Time Forecast Plot", x_lab="Date", y_lab="Value", interactive=True):
    # Convert date column to datetime if it's not already
    df[date_col] = pd.to_datetime(df[date_col])

    # Create interactive or static plot
    if interactive:
        fig = make_subplots()

        # Add trace for ACTUAL data
        actual_data = df[df['model_desc'] == 'ACTUAL']
        fig.add_trace(go.Scatter(x=actual_data[date_col], y=actual_data[value_col],
                                 mode='lines', name='ACTUAL',
                                 line=dict(width=2, dash='solid')))

        # Add traces for each model's predictions with confidence intervals
        for model_id in df['model_id'].unique():
            if model_id != 'Actual':
                model_data = df[df['model_id'] == model_id]
                fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data[value_col],
                                         mode='lines', name=model_data['model_desc'].iloc[0],
                                         line=dict(width=2, dash='dash')))
                # Confidence intervals
                fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data['conf_lo'],
                                         line=dict(width=0), showlegend=False, fill=None))
                fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data['conf_hi'],
                                         fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                                         line=dict(width=0), showlegend=False))

        fig.update_layout(title=title, xaxis_title=x_lab, yaxis_title=y_lab)
        fig.show()

    ##
    else:
        plt.figure(figsize=(12, 6))  # Adjusted for better visibility

        # Plot ACTUAL data
        actual_data = df[df['model_desc'] == 'ACTUAL']
        plt.plot(actual_data[date_col], actual_data[value_col], label='ACTUAL',
                 linewidth=2, linestyle='-')

        # Plot each model's predictions with confidence intervals
        for model_id in df['model_id'].unique():
            if model_id != 'Actual':
                model_data = df[df['model_id'] == model_id]
                plt.plot(model_data[date_col], model_data[value_col], label=model_data['model_desc'].iloc[0],
                         linewidth=2, linestyle='--')
                plt.fill_between(model_data[date_col], model_data['conf_lo'], model_data['conf_hi'],
                                 color='grey', alpha=0.3)

        # Determine the date frequency (yearly, quarterly, monthly)
        date_freq = pd.infer_freq(df[date_col])

        if date_freq is not None and date_freq.startswith('Q'):
            # Quarterly data
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        elif date_freq in ['M', 'MS']:
            # Monthly data
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 1990
        else:
            # Default formatting
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

        plt.gcf().autofmt_xdate()  # Auto-rotate for better readability

        plt.title(title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.legend()
        plt.show()

