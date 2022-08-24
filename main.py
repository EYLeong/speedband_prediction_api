from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import threading
from pathlib import Path
import time
import schedule

import config
import utils
import api_classes

Path(config.OUTPUT_PATH).unlink(missing_ok=True)
Path(config.LATEST_TIMESTEP_PATH).unlink(missing_ok=True)
Path(config.INPUT_SAMPLE_PATH).unlink(missing_ok=True)
utils.setup()


def threaded_job():
    def job():
        try:  # continue to run if datamall goes on maintenance or the server loses internet connection
            utils.fetch_all()
            utils.update_input()
            utils.predict()
        except:
            pass

    threading.Thread(target=job, daemon=True).start()


def scheduler():
    schedule.every(config.QUERY_INTERVAL).minutes.do(threaded_job)
    while True:
        schedule.run_pending()
        time.sleep(1)


threading.Thread(target=scheduler, daemon=True).start()

app = FastAPI()


@app.get("/", response_model=api_classes.PredictionsResponse)
async def root():
    return api_classes.PredictionsResponse.parse_file(config.OUTPUT_PATH)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Singapore Downtown Traffic Prediction",
        version="0.1.0",
        description="This API predicts the next 30 minutes of traffic speeds in the Singapore downtown area in 5 minute intervals. It uses traffic speed bands from DataMall and Deep Learning to make predictions.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
