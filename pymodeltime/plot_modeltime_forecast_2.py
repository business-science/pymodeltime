import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys
import math

def generate_distinct_colors(num_colors):
    """Generate a wider spectrum of distinct colors."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5 + 0.4 * (i % 2)  # Alternating between lighter and darker shades
        saturation = 0.7 + 0.3 * ((i + 1) % 2)  # Alternating saturation
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def plot_modeltime_forecast_2(df, date_col='date', value_col='value', group_col='Dept', key_col='key',
                              conf_lo_col='conf_lo', conf_hi_col='conf_hi',
                              title="Model Time Forecast Plot", y_lab="Value",
                              multi_column_display=False, include_depts=None):
    df[date_col] = pd.to_datetime(df[date_col])  # Ensure the date column is datetime

    if include_depts is None:
        include_depts = df[group_col].unique()

    # Determine if we are plotting predictions or future forecasts
    is_prediction = 'prediction' in df[key_col].values

    if is_prediction:
        # Filter departments that have prediction data
        depts = df[df[key_col] == 'prediction'][group_col].unique()
    else:
        # Use all departments for future forecasts
        depts = include_depts

    if not depts.size:
        print("No data available to plot.")
        return None

    rows = math.ceil(len(depts) / 2)  # Calculate the number of rows for two columns
    cols = 2 if len(depts) > 1 else 1
    subplot_titles = [str(dept) for dept in depts]  # Convert dept names to string
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, vertical_spacing=0.1,
                        subplot_titles=subplot_titles)

    unique_model_ids = df['model_id'].unique()
    color_palette = generate_distinct_colors(len(unique_model_ids))

    for i, dept in enumerate(depts, start=1):
        row = ((i - 1) // 2) + 1
        col = 1 if i % 2 != 0 else 2

        dept_df = df[df[group_col] == dept]
        actual_data = dept_df[dept_df[key_col] == 'actual']
        forecast_data = dept_df[dept_df[key_col] != 'actual']

        if not actual_data.empty:
            fig.add_trace(go.Scatter(x=actual_data[date_col], y=actual_data[value_col],
                                     mode='lines', name=f"{dept} - Actual",
                                     line=dict(color='blue', width=2)),
                          row=row, col=col)

        for j, model_id in enumerate(unique_model_ids):
            model_data = forecast_data[forecast_data['model_id'] == model_id]
            if model_data.empty:
                continue
            model_desc = model_data['model_desc'].iloc[0]
            color = color_palette[j]
            fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data[value_col],
                                     mode='lines', name=f"{dept} - {model_desc}",
                                     line=dict(color=color)),
                          row=row, col=col)
            # Add confidence interval traces
            fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data[conf_lo_col],
                                     mode='lines', name="Conf. Int. Low",
                                     line=dict(width=0),
                                     showlegend=False),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data[conf_hi_col],
                                     mode='lines', fill='tonexty', name="Conf. Int. High",
                                     line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.3)',
                                     showlegend=False),
                          row=row, col=col)

    
    fig.update_layout(title=title, title_x=0.3, margin=dict(l=40, r=40, t=40, b=40),
                      legend=dict(orientation="v", x=1.05, y=1, xanchor="left", yanchor="top"),
                      showlegend=True)

    # Set x-axis titles for the bottom plots
    for i in range(cols):
        fig.update_xaxes(title_text=date_col, row=rows, col=i+1)

    return fig

