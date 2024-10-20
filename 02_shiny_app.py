# BUSINESS SCIENCE UNIVERSITY
# WALMART FORECASTING DAILY DEMAND   
# PART 2: Shiny App
# ----

# GOAL: Predict Employee Churn (Attrition) 

# INSTRUCTIONS:
# shiny run --reload 02_shiny_app.py --port 8003
# Ctrl + C to shut down

# IMPORTS 
from shiny import (
    App, ui, reactive, Session, render
)
from shinywidgets import (
    output_widget, register_widget
)
import shinyswatch

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycaret.regression as reg
from pathlib import Path
import io
from datetime import datetime

from utils.timeseries_helpers import forecast_single_timeseries

# DIRECTORY PATHS ----

data_dir = Path(__file__).parent / "data"
www_dir = Path(__file__).parent / "www"

# DEFAULT APP INPUTS ----

TITLE = "Walmart Forecasting Daily Demand App"
TAB_1 = "Demand Forecaster"

PRODUCT_SELECTED = 'FOODS_3_090'
FORECAST_HORIZON = 365
MODEL_SELECTED   = 'xgboost'

# DATA & MODEL PREP ----
walmart_sales_raw_df = pd.read_csv(data_dir / "walmart_sales.csv")

unique_ids = walmart_sales_raw_df['item_id'].unique()

# model_choices = reg.models().index.tolist()
model_choices = ['lr',
 'lasso',
 'ridge',
 'en',
 'lar',
 'llar',
 'omp',
 'br',
 'ard',
 'par',
 'ransac',
 'tr',
 'huber',
 'kr',
 'svm',
 'knn',
 'dt',
 'rf',
 'et',
 'ada',
 'gbr',
 'mlp',
 'xgboost',
 'lightgbm',
 'catboost',
 'dummy']

# LAYOUT ----
page_dependencies = ui.tags.head(
    ui.tags.link(
        rel="stylesheet", type="text/css", href="style.css"
    )
)

app_ui = ui.page_navbar(
    
    # Bootswatch Themes: https://bootswatch.com/
    shinyswatch.theme.lux(),
    
    ui.nav(
        TAB_1,
        ui.layout_sidebar(
            sidebar=ui.panel_sidebar(
                
                ui.h3("Select A Product To Forecast"),
                
                ui.input_selectize(
                    "product_selected", 
                    "Product Selected:",
                    list(np.sort(unique_ids)),
                    selected=PRODUCT_SELECTED,
                    multiple=False
                ),
                
                ui.hr(),
                
                ui.h3('Forecast Parameters'),
                
                # ui.p("Forecast Horizon (Days)"),
                
                ui.input_numeric(
                    "forecast_horizon",
                    label = "Forecast Horizon (Days):",
                    value = FORECAST_HORIZON,
                    min   = 1,
                    max   = 365 * 2
                ), 
                
                # ui.p("Model Type"),
                
                ui.input_selectize(
                    "model_selected", 
                    "Model Selected:",
                    list(model_choices),
                    selected=MODEL_SELECTED,
                    multiple=False
                ),
                
                ui.hr(),
                
                ui.input_action_button(
                    "submit", "Submit", 
                    class_="btn-info"
                ),
                ui.download_button(
                    "download_forecast", "Download Forecast", class_="btn-info", style="margin:3px;"),
                
                width=3,
                # class_ = "well-gray",
            ),
            main = ui.panel_main(
                               
                ui.output_ui("ui_forecast_summary"),
                
                ui.column(
                    12,                    
                    ui.div(
                        output_widget("ui_forecast_plot") ,
                        class_="card",
                        style="margin:10px;"
                    )
                                      
                ),
                
            )
        ),
        
        
    ),
    title=ui.tags.div(
        #ui.img(src="business-science-logo.png", height="50px", style="margin:5px;"),
        ui.h4(" " + TITLE, style="color:white;margin-top:auto; margin-bottom:auto;"), 
        style="display:flex;-webkit-filter: drop-shadow(2px 2px 2px #222);"
    ),
    bg="#0062cc",
    inverse=True,
    header=page_dependencies
)

def server(input, output, session: Session):
    
    print("Server Started")
    
    # * Reactive data ----
    
    forecast_df = reactive.Value(
        forecast_single_timeseries(
            data      = walmart_sales_raw_df,
            id        = PRODUCT_SELECTED,
            h         = FORECAST_HORIZON,
            estimator = MODEL_SELECTED,
        )
    )
    
    # * Forecast on Submit ----
    
    @reactive.Effect
    @reactive.event(input.submit)
    def _1():
        
        df = forecast_single_timeseries(
            data      = walmart_sales_raw_df,
            id        = input.product_selected(),
            h         = input.forecast_horizon(),
            estimator = input.model_selected(),
        )
        
        print(df)
        
        forecast_df.set(df)
        
    @reactive.Effect
    def _2():
        
        fig = px.line(
            forecast_df(),
            y          = 'value',
            line_group = 'item_id',
            color      = 'key',
            color_discrete_sequence=[
                # 'rgb(247,251,255)',
                # 'rgb(222,235,247)',
                # 'rgb(198,219,239)',
                # 'rgb(158,202,225)',
                # 'rgb(107,174,214)',
                #'rgb(66,146,198)',
                'rgb(33,113,181)', #this
                # 'rgb(8,81,156)',
                # 'rgb(8,48,107)',
                'rgb(255, 0, 0)', #this
            ],
        )
        
        fig.update_layout(plot_bgcolor="white", hovermode="x unified")

        fig.update_xaxes(showline=False, gridcolor="#d2d2d2", gridwidth=0.5)
        fig.update_yaxes(showline=False, gridcolor="#d2d2d2", gridwidth=0.5)
        
        register_widget("ui_forecast_plot", go.FigureWidget(fig))
    
    @output
    @render.ui
    def ui_forecast_summary():
        
        median_demand = round(
            forecast_df().query('key == "FUTURE"')['value'].median()
        )
        
        html_out = ui.HTML(f"<h2>Product: {input.product_selected()} <br><small><span style='color:#3771b4;'>Median {input.forecast_horizon()}-Day Demand: {median_demand} Units</span></small></h2>")
        
        return html_out
    
    @session.download(
         filename=lambda: f"demand-forecast-{input.product_selected()}-{input.model_selected()}-{datetime.now().isoformat()}.csv"
    )
    def download_forecast():
        
        buf = io.StringIO()
        
        forecast_df() \
            .query('key == "FUTURE"') \
            .drop(['index.num', 'year', 'quarter', 'month', 'mday', 'week', 'wday'], axis=1) \
            .assign(model = lambda x: input.model_selected()) \
            .to_csv(buf, index=True, index_label="date")
        
        yield buf.getvalue()
    
    


app = App(
    app_ui, server, 
    static_assets=www_dir, 
    debug=False
)
