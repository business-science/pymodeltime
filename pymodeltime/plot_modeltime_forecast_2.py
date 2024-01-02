import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_modeltime_forecast_2(df, date_col='date', value_col='value', group_col='Dept', key_col='key',
                            title="Model Time Forecast Plot", y_lab="Value", 
                            interactive=True, multi_column_display=False, include_depts=None):
    df[date_col] = pd.to_datetime(df[date_col])  # Ensure the date column is datetime

    if interactive:
        if include_depts is None and 'prediction' in df[key_col].values:
            include_depts = df[df[key_col] == 'prediction'][group_col].unique()

        depts = df[group_col].unique() if include_depts is None else include_depts
        rows = len(depts) if multi_column_display else 1
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        for i, dept in enumerate(depts, start=1):
            dept_df = df[df[group_col] == dept]
            actual_data = dept_df[dept_df[key_col] == 'actual']
            predicted_data = dept_df[dept_df[key_col] == 'prediction']
            future_data = dept_df[dept_df[key_col] == 'future']

            if actual_data.empty and predicted_data.empty and future_data.empty:
                continue

            if not actual_data.empty:
                fig.add_trace(go.Scatter(x=actual_data[date_col], y=actual_data[value_col],
                                         mode='lines', name=f"{dept} - Actual",
                                         line=dict(color='blue', width=2)),
                              row=i, col=1)

            if not predicted_data.empty:
                for model_id in predicted_data['model_id'].unique():
                    model_data = predicted_data[predicted_data['model_id'] == model_id]
                    model_desc = model_data['model_desc'].iloc[0]
                    fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data[value_col],
                                             mode='lines', name=f"{dept} - {model_desc}",
                                             line=dict(dash='dash')),
                                  row=i, col=1)

            if not future_data.empty:
                for model_id in future_data['model_id'].unique():
                    model_data = future_data[future_data['model_id'] == model_id]
                    model_desc = model_data['model_desc'].iloc[0]
                    fig.add_trace(go.Scatter(x=model_data[date_col], y=model_data[value_col],
                                             mode='lines', name=f"{dept} - {model_desc}",
                                             line=dict(dash='dash', color='red')),
                                  row=i, col=1)

        fig.update_yaxes(title_text=y_lab, row='all', col=1)
        fig.update_layout(title=title, title_x=0.25, # Correctly center the title
                          legend=dict(orientation="v", x=1.05, y=1, xanchor="left", yanchor="top"),
                          showlegend=True)

        return fig
