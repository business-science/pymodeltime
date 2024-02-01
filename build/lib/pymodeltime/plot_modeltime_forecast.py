import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_modeltime_forecast(df, date_col='date', value_col='value',
                            title="Model Time Forecast Plot", x_lab="Date", y_lab="Value", interactive=True):
    if interactive:
        fig = make_subplots()

        # Plot actual data
        actual_data = df[df['model_desc'] == 'ACTUAL']
        fig.add_trace(go.Scatter(x=actual_data[date_col], y=actual_data[value_col],
                                 mode='lines', name='ACTUAL',
                                 line=dict(color='blue', width=2)))

        # Plot each model's predictions with confidence intervals
        model_ids = df['model_id'].unique()
        for model_id in model_ids:
            if model_id != 'Actual':
                model_data = df[df['model_id'] == model_id]
                model_desc = model_data['model_desc'].iloc[0]

                # Main prediction line
                fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data[value_col],
                                         mode='lines', name=model_desc,
                                         line=dict(dash='dash')))

                # Confidence intervals
                fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data['conf_lo'],
                                         mode='lines', name=f"{model_desc} Conf. Int. Low",
                                         line=dict(width=0),
                                         showlegend=False))
                fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data['conf_hi'],
                                         mode='lines', name=f"{model_desc} Conf. Int. High",
                                         fill='tonexty', line=dict(width=0),
                                         fillcolor='rgba(68, 68, 68, 0.3)',
                                         showlegend=False))

        # Update layout for interactive plot with centered title
        fig.update_layout(title=title, title_x=0.3,  # Center the title
                          xaxis_title=x_lab, yaxis_title=y_lab,
                          legend=dict(orientation="v", x=1.05, y=1, xanchor="left", yanchor="top"))
        return fig
    else:
        plt.figure(figsize=(12, 6))  # Adjusted for better visibility

        # Plot ACTUAL data
        actual_data = df[df['model_desc'] == 'ACTUAL']
        plt.plot(actual_data[date_col], actual_data[value_col], label='ACTUAL',
                 color='blue', linewidth=2, linestyle='-')

        # Plot each model's predictions with confidence intervals
        for model_id in df['model_id'].unique():
            if model_id != 'Actual':
                model_data = df[df['model_id'] == model_id]
                plt.plot(model_data[date_col], model_data[value_col], label=model_data['model_desc'].iloc[0],
                         linewidth=2, linestyle='--')
                plt.fill_between(model_data[date_col], model_data['conf_lo'], model_data['conf_hi'],
                                 color='grey', alpha=0.3)

        # Date formatting
        date_freq = pd.infer_freq(df[date_col])
        if date_freq is not None and date_freq.startswith('Q'):
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        elif date_freq in ['M', 'MS']:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        else:
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

        plt.gcf().autofmt_xdate(rotation=0)  # Set rotation to 0 for horizontal labels
        plt.title(title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend to the right side of the plot
        plt.show()



