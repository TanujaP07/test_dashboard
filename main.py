import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

with open('problem1.json') as problem:
    problem_1 = problem.read()

problem1_json = json.loads(problem_1)

confusion_matrices = np.array(problem1_json['confusion_matrices'])
radar_data = np.array(problem1_json['radar_data'])

losses_df = pd.DataFrame()

for i in range(len(problem1_json['losses'])):
    col_name = 'Model {}'.format(i + 1)
    losses_df[col_name] = problem1_json['losses'][i]

model_labels = list(losses_df.columns)
class_labels = ['Class 1', 'Class 2']
metric_labels = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5']

# Create traces for each model
traces = []
for i, model_label in enumerate(model_labels):
    trace = go.Scatterpolar(r=radar_data[i, :], theta=metric_labels, fill='toself', name=model_label)
    traces.append(trace)

# Create the layout for the radar chart
layout = go.Layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1]  # Adjust the range as needed
)), showlegend=True, title='Radar Chart for 5 Models and 5 Metrics')

# Create the radar chart figure
radar_fig = go.Figure(data=traces, layout=layout)

# Calculate accuracies
accuracies = []
for matrix in confusion_matrices:
    acc = matrix.diagonal() / matrix.sum(axis=1)
    accuracies.append(acc)

# Create a DataFrame for Plotly
data = {'Model': np.repeat(model_labels, 2), 'Class': np.tile(class_labels, len(model_labels)),
        'Accuracy': np.array(accuracies).flatten()}

# Create a bar plot with Plotly
bar_fig = px.bar(data, x='Model', y='Accuracy', color='Model', title='Classwise Accuracies for 5 Models',
                 labels={'Model': 'Model', 'Accuracy': 'Accuracy', 'Class': 'Class'},
                 category_orders={'Class': class_labels}, color_discrete_sequence=px.colors.qualitative.Set1)

app = Dash(__name__)

app.layout = html.Div([

    html.H1('Model Evaluation'),

    html.Div([dcc.Graph(id='bar-chart', figure=bar_fig, style={'width': '49%', 'display': 'inline-block'}),
              dcc.Graph(id='radar-metric', figure=radar_fig, style={'width': '49%', 'display': 'inline-block'})]),

    dcc.Dropdown(id='model-selector', options=[{'label': m, 'value': m} for m in model_labels], value=[model_labels[0]],
                 multi=True), dcc.Graph(id='losses-plot'), html.P('Smoothing'),
    dcc.Slider(id='smoothing', min=0, max=1, value=0.5, step=0.1)

])


@app.callback(Output('losses-plot', 'figure'), [Input('model-selector', 'value'), Input('smoothing', 'value')])
def update_losses(selected_models, smoothing):
    traces = []

    # Generate trace for each selected model
    for col in selected_models:
        # Fit model
        model = SimpleExpSmoothing(losses_df[col])
        fitted = model.fit(smoothing_level=smoothing, optimized=False)

        # Create trace
        trace = go.Scatter(x=losses_df.index, y=fitted.fittedvalues, mode='lines', name=col)

        traces.append(trace)

    return {'data': traces,
        'layout': {'title': 'Model Losses', 'xaxis': {'title': 'Steps'}, 'yaxis': {'title': 'Loss'}}}


if __name__ == '__main__':
    app.run_server(debug=True)
