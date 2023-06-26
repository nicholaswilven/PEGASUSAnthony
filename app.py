from model.generate_demo import abs_summary, generate_from_index

from dash import Dash, html, callback, Output, Input

from fastapi import FastAPI

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1("Machine Learning Model Demo"),
        html.Textarea(id="article", placeholder="Enter the article text here..."),
        html.Div(
            id="summary",
            children=[
                html.H3("Alternative Summaries:"),
                html.Ol(id="summary-list"),
                html.Button("Generate Summaries", id="generate-button"),
            ],
        ),
    ]
)

@callback(
    Output("summary-list", "children"),
    [Input("generate-button", "n_clicks")],
    [State("article", "value")],
)
def generate_summaries(n_clicks, article):
    if n_clicks is None:
        return []
    
    summaries = get_alternative_summaries(article)
    return [html.Li(summary) for summary in summaries]


def get_alternative_summaries(article):
    return abs_summary(article)['result']

if __name__ == "__main__":
    app.run_server(debug=True)
