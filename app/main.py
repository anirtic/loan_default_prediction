import io
import os
import config
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from data_extraction.data_extractor_pipe import (
    DataExtractor,
    SequentialDataPipeline,
    ColumnDropper,
    ContractTypeSelector,
    ScaleMinMax,
    RemoveInf,
    ColumnRenamer,
    CategoricalColsConverter,
    FeatureSelection,
    ModelLoader,
    get_preprocessing_pipeline
)

app = FastAPI()


async def extract_input_data(request, file: UploadFile=None) -> pd.DataFrame:
    """
    Attempts to extract input data from a file upload or a JSON body in an HTTP request.

    Parameters:
    - request: The HTTP request object.
    - file (UploadFile, optional): An uploaded file.

    Returns:
    - pd.DataFrame: The extracted input data as a DataFrame.

    Raises:
    - HTTPException: If neither a valid file nor valid JSON content can be processed.
    """
    if file:
        contents = await file.read()
        input_data = pd.read_csv(io.BytesIO(contents))
    else:
        try:
            contents = await request.json()
            input_data = pd.DataFrame([contents])
            input_data.replace('', np.nan, inplace=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request content: {str(e)}")
    return input_data


@app.post("/default_prediction")
async def model_default(request: Request, file: UploadFile = File(None)):

    input_data = await extract_input_data(request, file)
    preprocessing_pipeline = get_preprocessing_pipeline(application_df=input_data)

    default_pipeline = Pipeline(
        steps=[
            ("preprocessing_pipeline", preprocessing_pipeline),
            ("scaler", ScaleMinMax(scaler="default_scaler")),
            ("col_renamer", ColumnRenamer()),
            ("feature_selector", FeatureSelection("rfecv_default_lgbm_ver2")),
            ("model", ModelLoader(name="default_model_lgb"))
        ]
    )
    y_pred_def = default_pipeline.predict(input_data)
    response_data = float(y_pred_def)
    return jsonable_encoder(response_data)


@app.post("/early_repayment_prediction")
async def model_early_repayment(request: Request, file: UploadFile = File(None)):

    input_data = await extract_input_data(request, file)
    preprocessing_pipeline = get_preprocessing_pipeline(
        application_df=input_data,
        drop_columns=["SK_ID_CURR", "NAME_CONTRACT_TYPE", "POS_COMPLETED_BEFORE_TERM"]
    )

    early_pymnt_pipeline = Pipeline(
        steps=[
            ("preprocessing_pipeline", preprocessing_pipeline),
            ("scaler", ScaleMinMax(scaler="early_paymnt_scaler")),
            ("col_renamer", ColumnRenamer()),
            ("feature_selector", FeatureSelection("rfecv_early_pymnt")),
            ("model", ModelLoader(name="early_payment_model_lgb"))
        ]
    )
    y_pred_early = early_pymnt_pipeline.predict(input_data)
    response_data = float(y_pred_early)
    return jsonable_encoder(response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/")
async def root():
    return {"message": "Default and early repayment behaviour prediction"}
