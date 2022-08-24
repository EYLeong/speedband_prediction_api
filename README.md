# Speedband Prediction API
This is the source code for the Singapore traffic speedband prediction API

## Installation
Install [Anaconda](https://www.anaconda.com/products/distribution)

Recreate the environment with `conda create -f env.yml`

Activate the environment with `conda activate speedbands`

## Setup
Get a DataMall API key [here](https://datamall.lta.gov.sg/content/datamall/en/request-for-api.html)

Copy and paste the key into the `API_KEY` variable in __config.py__

## User Guide
Create __/processed__, __/raw__, and __/temp__ folders in the __data__ folder

Start the API with `uvicorn main:app`

The API works by querying DataMall every 5 minutes to generate a sample of the past 1 hour of traffic speeds, and using that sample to generate predictions for the next 30 minutes. Since there needs to be a minimum of 12 input timesteps to make a prediction, no predictions will be available for the first 60 minutes and querying the root of the API will return a 500 response. Once enough data has been collected, the root will instead return the predictions in a format according to the docs available from the `/docs` route. This API is based on the OpenAPI spec, and the schema is available from the `/openapi.json` route.

By default, data more than 1 hour in the past is deleted. To save all collected data, set `SAVE_RAW` in __config.py__ to `True`

