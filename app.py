from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run
from typing import Optional

from CustomerChurn.constants import APP_HOST, APP_PORT
from CustomerChurn.pipline.prediction_pipeline import ChurnInputData, ChurnPredictor
from CustomerChurn.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adapté au formulaire de churn
class ChurnForm:
    def __init__(self, request: Request):
        self.request: Request = request
        # Tous les champs que tu attends depuis le formulaire HTML
        self.gender: Optional[str] = None
        self.SeniorCitizen: Optional[int] = None
        self.Partner: Optional[str] = None
        self.Dependents: Optional[str] = None
        self.tenure: Optional[int] = None
        self.PhoneService: Optional[str] = None
        self.MultipleLines: Optional[str] = None
        self.InternetService: Optional[str] = None
        self.OnlineSecurity: Optional[str] = None
        self.OnlineBackup: Optional[str] = None
        self.DeviceProtection: Optional[str] = None
        self.TechSupport: Optional[str] = None
        self.StreamingTV: Optional[str] = None
        self.StreamingMovies: Optional[str] = None
        self.Contract: Optional[str] = None
        self.PaperlessBilling: Optional[str] = None
        self.PaymentMethod: Optional[str] = None
        self.MonthlyCharges: Optional[float] = None
        self.TotalCharges: Optional[float] = None

    async def get_churn_data(self):
        form = await self.request.form()
        self.gender = form.get("gender")
        self.SeniorCitizen = int(form.get("SeniorCitizen"))
        self.Partner = form.get("Partner")
        self.Dependents = form.get("Dependents")
        self.tenure = int(form.get("tenure"))
        self.PhoneService = form.get("PhoneService")
        self.MultipleLines = form.get("MultipleLines")
        self.InternetService = form.get("InternetService")
        self.OnlineSecurity = form.get("OnlineSecurity")
        self.OnlineBackup = form.get("OnlineBackup")
        self.DeviceProtection = form.get("DeviceProtection")
        self.TechSupport = form.get("TechSupport")
        self.StreamingTV = form.get("StreamingTV")
        self.StreamingMovies = form.get("StreamingMovies")
        self.Contract = form.get("Contract")
        self.PaperlessBilling = form.get("PaperlessBilling")
        self.PaymentMethod = form.get("PaymentMethod")
        self.MonthlyCharges = float(form.get("MonthlyCharges"))
        self.TotalCharges = float(form.get("TotalCharges"))


@app.get("/", tags=["form"])
async def index(request: Request):
    return templates.TemplateResponse("churn.html", {"request": request, "context": "Remplissez le formulaire"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Modèle entraîné avec succès !")
    except Exception as e:
        return Response(f"Erreur pendant l'entraînement : {e}")


@app.post("/")
async def predictChurnClient(request: Request):
    try:
        form = ChurnForm(request)
        await form.get_churn_data()

        churn_data = ChurnInputData(
            gender=form.gender,
            SeniorCitizen=form.SeniorCitizen,
            Partner=form.Partner,
            Dependents=form.Dependents,
            tenure=form.tenure,
            PhoneService=form.PhoneService,
            MultipleLines=form.MultipleLines,
            InternetService=form.InternetService,
            OnlineSecurity=form.OnlineSecurity,
            OnlineBackup=form.OnlineBackup,
            DeviceProtection=form.DeviceProtection,
            TechSupport=form.TechSupport,
            StreamingTV=form.StreamingTV,
            StreamingMovies=form.StreamingMovies,
            Contract=form.Contract,
            PaperlessBilling=form.PaperlessBilling,
            PaymentMethod=form.PaymentMethod,
            MonthlyCharges=form.MonthlyCharges,
            TotalCharges=form.TotalCharges,
        )

        input_df = churn_data.get_input_data_frame()
        model_predictor = ChurnPredictor()
        prediction = model_predictor.predict(input_df)[0]

        status = "Client en risque de churn" if prediction == 1 else "Client fidèle (pas de churn)"

        return templates.TemplateResponse("churn.html", {"request": request, "context": status})

    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)