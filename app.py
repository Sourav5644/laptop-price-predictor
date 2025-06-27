from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import LaptopData, LaptopDataRegressor
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Company: Optional[object] = None
        self.TypeName: Optional[object] = None
        self.Ram: Optional[int] = None
        self.Weight: Optional[float] = None
        self.Touchscreen: Optional[int] = None
        self.IPS: Optional[int] = None
        self.cpu_name: Optional[object] = None
        self.SSD: Optional[int] = None
        self.HDD: Optional[int] = None
        self.gpu_brand: Optional[object] = None
        self.os: Optional[object] = None

    async def get_Laptop_data(self):
        form = await self.request.form()
        self.Company = form.get("Company")
        self.TypeName = form.get("TypeName")
        self.Ram = int(form.get("Ram"))
        self.Weight = float(form.get("Weight"))
        self.Touchscreen = int(form.get("Touchscreen"))
        self.IPS = int(form.get("IPS"))
        self.cpu_name = form.get("cpu_name")
        self.SSD = int(form.get("SSD"))
        self.HDD = int(form.get("HDD"))
        self.gpu_brand = form.get("gpu_brand")
        self.os = form.get("os")

@app.get("/", tags=["UI"])
async def index(request: Request):
    return templates.TemplateResponse(
        "Laptopdata.html", {"request": request, "context": "Enter laptop details for price prediction"}
    )

@app.get("/train", tags=["Training"])
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/", tags=["Prediction"])
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_Laptop_data()

        Laptop_data = LaptopData(
            Company=form.Company,
            TypeName=form.TypeName,
            Ram=form.Ram,
            Weight=form.Weight,
            Touchscreen=form.Touchscreen,
            IPS=form.IPS,
            cpu_name=form.cpu_name,
            gpu_brand=form.gpu_brand,
            os=form.os,
            SSD=form.SSD,
            HDD=form.HDD
        )

        Laptop_df = Laptop_data.get_laptop_input_data_frame()

        model_predictor = LaptopDataRegressor()
        predicted_price = model_predictor.predict(dataframe=Laptop_df)[0]

        # Format price output nicely
        result = f"Predicted Price: ₹{int(predicted_price):,}"

        return templates.TemplateResponse(
            "Laptopdata.html",
            {"request": request, "context": result},
        )

    except Exception as e:
        return templates.TemplateResponse(
            "Laptopdata.html",
            {"request": request, "context": f"Error: {e}"},
        )

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
