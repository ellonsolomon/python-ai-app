import dash  
from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table
from io import StringIO


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display':'grid', 'grid-auto-flow': 'row'}
row_style = {'display':'grid', 'grid-auto-flow': 'column'}

import plotly.express as px
import pandas as pd

import requests

app = Dash(__name__)

BASE_URL = " http://172.28.175.67:4000"


df = pd.read_csv("iris_extended_encoded.csv",sep=',')
df_csv = df.to_csv(index=False)

app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Tabs([
    dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

    html.Div([
        html.Div([
            html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
            dcc.Input(id='file-for-train', type='text', style={'width':'100px'}),
            html.Div([
                html.Button('Load', id='load-val', style={"width":"60px", "height":"30px"}),
                html.Div(id='load-response', children='Click to load')
            ], style=col_style)
        ], style=col_style),

        html.Div([
            html.Button('Upload', id='upload-val', style={"width":"60px", "height":"30px"}),
            html.Div(id='upload-response', children='Click to upload')
        ], style=col_style| {'margin-top':'20px'})

    ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),


html.Div([
    html.Div([
        html.Div([
            html.Label(['Feature'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                df.columns, 
                df.columns[0],          #<default value for dropdown>
                id='hist-column'
            )
            ], style=col_style ),
        dcc.Graph( id='selected_hist' )
    ], style=col_style | {'height':'400px', 'width':'400px'}),

    html.Div([

    html.Div([

    html.Div([
        html.Label(['X-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns, 
            df.columns[0],          #<default value for dropdown>
            id='xaxis-column'
            )
        ]),

    html.Div([
        html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns, 
            df.columns[0],         #<default value for dropdown>
            id='yaxis-column'
            )
        ])
    ], style=row_style | {'margin-left':'50px', 'margin-right': '50px'}),

    dcc.Graph(id='indicator-graphic')
    ], style=col_style)
], style=row_style),


    html.Div(id='tablecontainer', children=[
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
            id='datatable' )
        ])
    ]),
    dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Button('New model', id='build-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='build-response', children='Click to build new model and train')
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Re-Train', id='train-val', style={"width":"90px", "height":"30px"})
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-train', children='')
    ]),
    dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='row-for-score', type='text', style={'width':'300px'}))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-score', type='text'))
            ], style=col_style | {'margin-top':'20px'}),            
            html.Div([
                html.Button('Score', id='score-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='score-response', children='Click to score')
            ], style=col_style | {'margin-top':'20px'})
        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),
        
        html.Div(id='container-button-score', children='')
    ]),

    dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Test', id='test-val'),
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-test', children='')
    ])

    ])
])


@app.callback(
    Output(component_id='load-response', component_property='children'),
    Input(component_id='file-for-train', component_property='value'),
    Input(component_id='load-val', component_property='n_clicks')
)

def update_output_load(filename, nclicks):
    global df, df_csv

    if filename is not None and nclicks is not None:
        df = pd.read_csv(filename, sep=',')
        df_csv = df.to_csv(index=False)
        return 'Load done.'
    else:
        return ''


@app.callback(
    Output(component_id="build-response", component_property="children"),
    Input(component_id="dataset-for-train", component_property="value"),
    Input(component_id="build-val", component_property="n_clicks")
)
def update_output_build(dataset_ID,nclicks):
    print (nclicks)
    if nclicks != None:
        url = f"{BASE_URL}/iris/model"  
    # Data payload includes the dataset ID to be used  
        data = {'dataset': dataset_ID}  
    # Send a POST request with the dataset ID to build and train the model  
        response = requests.post(url, data=data)  
        return f'the model ID {response.text}'
    else:
        return ''



@app.callback(
    Output(component_id='upload-response', component_property='children'),
    Input(component_id='upload-val', component_property='n_clicks')
)
def update_output_upload(nclicks):
    global df_csv

    if nclicks != None:
        url = f"{BASE_URL}/iris/datasets"
        files = {'train': df_csv}
        response = requests.post(url, files=files)  
        
        return f'return the dataset ID {response.text}'
    else:
        return ''
    
@app.callback(
    Output(component_id="indicator-graphic", component_property="figure"),
    Input(component_id="xaxis-column", component_property="value"),
    Input(component_id="yaxis-column", component_property="value")
)

def update_graph(xaxis_column_name, yaxis_column_name):

    fig = px.scatter(x=df.loc[:, xaxis_column_name].values,
                     y=df.loc[:, yaxis_column_name].values)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=xaxis_column_name)
    fig.update_yaxes(title=yaxis_column_name)

    return fig


@app.callback(
    Output(component_id='selected_hist', component_property='figure'),
    Input(component_id='hist-column', component_property='value')
)


def update_hist(column_name):

    fig = px.histogram(df, x=column_name)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=column_name)

    return fig


@app.callback(
    Output(component_id='tablecontainer', component_property='children'),
    Input(component_id='load-val', component_property='n_clicks')
)

def update_table(nclicks):
    return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
        id='datatable')



@app.callback(
    Output(component_id='container-button-train', component_property='children'),
    Input(component_id="model-for-train", component_property="value"),
    Input(component_id="dataset-for-train", component_property="value"),
    Input(component_id="train-val", component_property="n_clicks")
)

def update_output_train(model_id, dataset_ID, nclicks):
    if nclicks != None:
        # add API endpoint request here
        url = f"{BASE_URL}/iris/model/{model_id}?dataset={dataset_ID}"  
        # Send a PUT request to re-train the model and get the response  
        response = requests.put(url) 
        
        train_df = pd.read_json(response.text)
        train_fig = px.line(train_df)

        return dcc.Graph( figure=train_fig )
    else:
        return ""


@app.callback(  
    Output(component_id='container-button-score', component_property='children'),  
    Input(component_id="model-for-score", component_property="value"),  
    Input(component_id="row-for-score", component_property="value"),  
    Input(component_id="score-val", component_property="n_clicks"))  
def update_output_score(model_ID, row, nclicks):  
    if nclicks is not None and model_ID and row:  
        try:  
            # Convert the input row from CSV string to a list of floats  
            features = [float(x.strip()) for x in row.split(',')]  
              
            # Construct the API request for scoring  
            url = f"{BASE_URL}/iris/model/{model_ID}"  
            params = {'fields': ','.join(map(str, features))}  
              
            # Send a GET request to get the model's score on the given features  
            response = requests.get(url, params=params)  
              
            # Check the response  
            if response.ok:  
                response_json = response.json()  
                # Use the key 'result' to retrieve the score  
                score_result = response_json.get('result')  
                  
                if score_result:  
                    return html.Div([  
                        html.P(f"Model scored, result: {score_result}"),  
                    ])  
                else:  
                    return html.Div([  
                        html.P("Score result not found in the response."),  
                    ])  
            else:  
                print(f"Failed to get a score. Status code: {response.status_code}. Response: {response.text}")  
                return html.Div([  
                    html.P("Failed to get a score from the model."),  
                ])  
        except ValueError as e:  
            return html.Div([  
                html.P(f"Error converting input to floats: {e}"),  
            ])  
    else:  
        return ""  



@app.callback(
    Output(component_id='container-button-test', component_property='children'),
    Input(component_id="model-for-test", component_property="value"),
    Input(component_id="dataset-for-test", component_property="value"),
    Input(component_id="test-val", component_property="n_clicks")
)


def update_output_test(model_ID, dataset_ID, nclicks):
    if nclicks != None:
        # add API endpoint request for testing with given dataset ID
        url = f"{BASE_URL}/iris/model/{model_ID}/test?dataset={dataset_ID}"  
        # Send a GET request to test the model and get the response  
        response = requests.get(url)  

        test_df = pd.read_json(StringIO(response.text))
        test_df = test_df.drop(columns=['confusion_matrix'])
        test_fig = px.line(test_df)

        return dcc.Graph( figure=test_fig )
    else:
        return ""

 
# if __name__ == '__main__':
#     app.run_server(debug=True)

if __name__ == '__main__':  
    app.run_server(debug=True, host='0.0.0.0', port=8050)  
