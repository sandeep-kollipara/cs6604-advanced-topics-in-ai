# *************** Main Workflow ***************

from agents.loader_agent import LoaderAgent
import agents.router_agent
#import panel as pn
#import pandas as pd
#import matplotlib.pyplot as plt
#import io
import warnings as wrn

wrn.filterwarnings('ignore')


#if '__main__'.__eq__(__name__):
'''
    user_input = input('\nTalk to the AI agent: ')
    nextAgent = LoaderAgent(None)
    prevAgent = nextAgent(user_input) # RouterAgent
    while nextAgent is not None:
        user_input = input('\nContinue talking to AI agent: ')
        nextAgent = prevAgent(user_input) # WorkerAgent unless user fast-forwards or rewinds
        if type(nextAgent) == agents.router_agent.RouterAgent or nextAgent is None:
            prevAgent = nextAgent
            continue
        else:
            prevAgent = nextAgent(user_input)
    print('\n*** Program End ***')
'''
'''
    pn.extension('tabulator')  # for nice DataFrame display

    # Sample DataFrame
    df = pd.DataFrame({
        'x': range(10),
        'y': [i**2 for i in range(10)]
    })

    # --- Chat Interface Components ---
    chat_history = pn.pane.Markdown("", sizing_mode="stretch_both")
    chat_input = pn.widgets.TextInput(placeholder="Type a message...", sizing_mode="stretch_width")

    def send_chat(event):
        if chat_input.value.strip():
            chat_history.object += f"\n\n**You:** {chat_input.value.strip()}"
            chat_input.value = ""

    chat_input.param.watch(send_chat, 'value', onlychanged=False)

    # --- DataFrame View ---
    dataframe_view = pn.widgets.Tabulator(df, height=300)

    # --- Plot View ---
    def create_plot(df):
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(df['x'], df['y'], marker='o')
        ax.set_title('Sample Plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        return pn.pane.PNG(buf.getvalue(), sizing_mode="stretch_both")

    plot_view = create_plot(df)

    # --- Layout ---
    left_panel = pn.Column(
        pn.pane.Markdown("### Chat", sizing_mode="stretch_width"),
        chat_history,
        chat_input,
        sizing_mode="stretch_both"
    )

    right_panel = pn.Column(
        pn.pane.Markdown("### DataFrame View", sizing_mode="stretch_width"),
        dataframe_view,
        pn.pane.Markdown("### Plot View", sizing_mode="stretch_width"),
        plot_view,
        sizing_mode="stretch_both"
    )

    layout = pn.Row(
        left_panel,
        right_panel,
        sizing_mode="stretch_both"
    )

    # --- Serve the app ---
    layout.servable()
'''
'''
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.express as px
import pandas as pd

# Example dataframe
df = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "Value": [100, 200, 300, 400]
})

# Create a simple plot
fig = px.bar(df, x="Category", y="Value", title="Sample Bar Plot")

# Start the app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H2("Chat Interface"),
        html.Div(id='chat-history', style={
            'height': '70vh', 'overflowY': 'scroll', 'border': '1px solid #ccc', 'padding': '10px',
            'backgroundColor': '#f9f9f9'
        }),
        dcc.Input(id='chat-input', type='text', placeholder='Type a message...', style={'width': '80%'}),
        html.Button('Send', id='send-button', n_clicks=0)
    ], style={
        'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'
    }),

    html.Div([
        html.Div([
            html.H2("DataFrame View"),
            dash_table.DataTable(
                id='data-table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
        ], style={'height': '45vh', 'overflowY': 'auto', 'marginBottom': '10px'}),

        html.Div([
            html.H2("Plot View"),
            dcc.Graph(
                id='plot',
                figure=fig
            )
        ], style={'height': '45vh'})
    ], style={
        'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'
    })
])

# Store chat history in a hidden div (or later could use dcc.Store)
app.clientside_callback(
    """
    function(n_clicks, value, history) {
        if (ctx.triggered.map(t => t.prop_id).includes('send-button.n_clicks')) {
            if (value && value.trim() !== '') {
                const updated_history = history || [];
                updated_history.push('User: ' + value);
                return [updated_history.map(m => m + '\\n').join(''), ''];
            }
        }
        return [history ? history.map(m => m + '\\n').join('') : '', value];
    }
    """,
    Output('chat-history', 'children'),
    Output('chat-input', 'value'),
    Input('send-button', 'n_clicks'),
    State('chat-input', 'value'),
    State('chat-history', 'children'),
)

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(debug=True)
'''

import dash
from dash import html, dcc, Output, Input, State, ctx
import dash_table
import pandas as pd
import plotly.express as px

# Create some initial dummy data
df = pd.DataFrame({
    'x': list(range(10)),
    'y': [i**2 for i in range(10)]
})

app = dash.Dash(__name__)
app.title = "Console-DataFrame-Plot App"

app.layout = html.Div([
    html.Div([
        html.H3("Console Interface"),
        dcc.Textarea(
            id='console-input',
            value='Type Python commands here...',
            style={'width': '100%', 'height': '300px'}
        ),
        html.Button('Run', id='run-button', n_clicks=0),
        html.Div(id='console-output', style={'whiteSpace': 'pre-wrap', 'marginTop': '20px'})
    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

    html.Div([
        html.Div([
            html.H3("DataFrame View"),
            dash_table.DataTable(
                id='dataframe-view',
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'}
            )
        ], style={'height': '50%', 'overflowY': 'auto'}),

        html.Div([
            html.H3("Plot View"),
            dcc.Graph(id='plot-view')
        ], style={'height': '50%'})
    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
])

# Application state (simulate variable sharing)
global_df = df.copy()

@app.callback(
    Output('dataframe-view', 'data'),
    Output('dataframe-view', 'columns'),
    Output('plot-view', 'figure'),
    Output('console-output', 'children'),
    Input('run-button', 'n_clicks'),
    State('console-input', 'value')
)
def run_console(n_clicks, console_code):
    global global_df
    if n_clicks == 0:
        # Initial rendering
        fig = px.line(global_df, x='x', y='y', title='Initial Plot')
        return global_df.to_dict('records'), [{"name": col, "id": col} for col in global_df.columns], fig, ""
    
    try:
        # Only allow editing the global_df (security!)
        local_vars = {'df': global_df.copy(), 'pd': pd}
        exec(console_code, {}, local_vars)
        if 'df' in local_vars:
            global_df = local_vars['df']
        
        fig = px.line(global_df, x=global_df.columns[0], y=global_df.columns[1], title='Updated Plot')
        output_message = "Executed successfully."
    except Exception as e:
        fig = px.line(global_df, x=global_df.columns[0], y=global_df.columns[1], title='Error Plot')
        output_message = f"Error: {str(e)}"
    
    return global_df.to_dict('records'), [{"name": col, "id": col} for col in global_df.columns], fig, output_message

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(debug=True)
