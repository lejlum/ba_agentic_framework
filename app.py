import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, callback, dcc, html
from dash_iconify import DashIconify


app = dash.Dash(__name__)
server = app.server

df = px.data.gapminder()


def create_scatter_plot(selected_year, selected_continent=None):
    filtered_df = df[df["year"] == selected_year]

    if selected_continent and selected_continent != "All":
        filtered_df = filtered_df[filtered_df["continent"] == selected_continent]

    fig = px.scatter(
        filtered_df,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
        title=f"Life Expectancy vs GDP per Capita ({selected_year})",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_line_chart(selected_country):
    country_data = df[df["country"] == selected_country]
    fig = px.line(
        country_data,
        x="year",
        y="lifeExp",
        title=f"{selected_country} - Life Expectancy",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_bar_chart(selected_year):
    year_data = df[df["year"] == selected_year]
    continent_stats = year_data.groupby("continent")["lifeExp"].mean().reset_index()
    fig = px.bar(
        continent_stats,
        x="continent",
        y="lifeExp",
        color="continent",
        title=f"Average Life Expectancy by Continent ({selected_year})",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def create_datacard(title, value, icon, color):
    return dmc.Card(
        [
            dmc.Group(
                [
                    DashIconify(icon=icon, width=30, color=color),
                    html.Div(
                        [
                            dmc.Text(value, size="xl", fw=700, c="white"),
                            dmc.Text(title, size="sm", c="dimmed"),
                        ]
                    ),
                ],
                align="center",
                gap="md",
            )
        ],
        p="md",
        className="datacard",
    )


app.layout = dmc.MantineProvider(
    [
        html.Link(
            href="https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap",
            rel="stylesheet",
        ),
        dmc.Group(
            [
                DashIconify(icon="twemoji:globe-with-meridians", width=45),
                dmc.Text(
                    "Gapminder World Data Explorer", ml=10, size="xl", fw=900, c="white"
                ),
            ],
            align="center",
            className="header",
            mb="md",
        ),
        dmc.Grid(
            [
                dmc.GridCol(
                    [
                        dmc.Stack(
                            [
                                dmc.Card(
                                    [
                                        dmc.Text("Controls", size="lg", mb="md"),
                                        dmc.Stack(
                                            [
                                                html.Div(
                                                    [
                                                        dmc.Text(
                                                            "Year:", size="sm", mb=5
                                                        ),
                                                        dmc.Slider(
                                                            id="year-slider",
                                                            min=1952,
                                                            max=2007,
                                                            step=5,
                                                            value=2007,
                                                            marks=[
                                                                {
                                                                    "value": year,
                                                                    "label": str(year),
                                                                }
                                                                for year in [
                                                                    1952,
                                                                    1967,
                                                                    1982,
                                                                    1997,
                                                                    2007,
                                                                ]
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        dmc.Text(
                                                            "Continent Filter:",
                                                            size="sm",
                                                            mb=5,
                                                        ),
                                                        dmc.Select(
                                                            id="continent-dropdown",
                                                            data=[
                                                                {
                                                                    "value": "All",
                                                                    "label": "All Continents",
                                                                }
                                                            ]
                                                            + [
                                                                {
                                                                    "value": cont,
                                                                    "label": cont,
                                                                }
                                                                for cont in sorted(
                                                                    df[
                                                                        "continent"
                                                                    ].unique()
                                                                )
                                                            ],
                                                            value="All",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        dmc.Text(
                                                            "Select Country:",
                                                            size="sm",
                                                            mb=5,
                                                        ),
                                                        dmc.Select(
                                                            id="country-dropdown",
                                                            data=[
                                                                {
                                                                    "value": country,
                                                                    "label": country,
                                                                }
                                                                for country in sorted(
                                                                    df[
                                                                        "country"
                                                                    ].unique()
                                                                )
                                                            ],
                                                            value="United States",
                                                            searchable=True,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            gap="lg",
                                        ),
                                    ],
                                    p="md",
                                    className="control-card",
                                )
                            ]
                        )
                    ],
                    span=3,
                ),
                dmc.GridCol(
                    [
                        dmc.Stack(
                            [
                                html.Div(id="stats-cards"),
                                dmc.Card(
                                    [dcc.Graph(id="scatter-plot")],
                                    p="sm",
                                    className="chart-card",
                                ),
                            ],
                            gap="md",
                        )
                    ],
                    span=9,
                ),
            ],
            gutter="md",
        ),
        dmc.Grid(
            [
                dmc.GridCol(
                    [
                        dmc.Card(
                            [dcc.Graph(id="line-chart")], p="sm", className="chart-card"
                        )
                    ],
                    span=6,
                ),
                dmc.GridCol(
                    [
                        dmc.Card(
                            [dcc.Graph(id="bar-chart")], p="sm", className="chart-card"
                        )
                    ],
                    span=6,
                ),
            ],
            gutter="md",
            mt="md",
        ),
    ],
    forceColorScheme="dark",
    theme={"colorScheme": "dark"},
)


@callback(
    Output("scatter-plot", "figure"),
    [Input("year-slider", "value"), Input("continent-dropdown", "value")],
)
def update_scatter_plot(selected_year, selected_continent):
    return create_scatter_plot(selected_year, selected_continent)


@callback(Output("line-chart", "figure"), Input("country-dropdown", "value"))
def update_line_chart(selected_country):
    return create_line_chart(selected_country)


@callback(Output("bar-chart", "figure"), Input("year-slider", "value"))
def update_bar_chart(selected_year):
    return create_bar_chart(selected_year)


@callback(Output("stats-cards", "children"), Input("year-slider", "value"))
def update_stats(selected_year):
    year_data = df[df["year"] == selected_year]

    avg_life_exp = round(year_data["lifeExp"].mean(), 1)
    total_pop = year_data["pop"].sum()
    num_countries = len(year_data)
    avg_gdp = round(year_data["gdpPercap"].mean(), 0)

    return dmc.Grid(
        [
            dmc.GridCol(
                create_datacard(
                    "Life Expectancy",
                    f"{avg_life_exp} years",
                    "mdi:heart-pulse",
                    "#ff6b35",
                ),
                span=3,
            ),
            dmc.GridCol(
                create_datacard(
                    "Population",
                    f"{total_pop / 1e9:.1f}B",
                    "mdi:account-group",
                    "#1f77b4",
                ),
                span=3,
            ),
            dmc.GridCol(
                create_datacard(
                    "Countries", str(num_countries), "mdi:earth", "#2ca02c"
                ),
                span=3,
            ),
            dmc.GridCol(
                create_datacard(
                    "GDP per Capita", f"${avg_gdp:,.0f}", "mdi:currency-usd", "#d62728"
                ),
                span=3,
            ),
        ],
        gutter="sm",
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
