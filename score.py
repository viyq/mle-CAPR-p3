# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# Modified by Victor Y.Q.

import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
import time

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Name": pd.Series(["example_value"], dtype="object"), "ID": pd.Series(["example_value"], dtype="object"), "MW": pd.Series([0.0], dtype="float64"), "log_Kow": pd.Series([0.0], dtype="float64"), "log_D": pd.Series([0.0], dtype="float64"), "dipole": pd.Series([0.0], dtype="float64"), "MV": pd.Series([0.0], dtype="float64"), "length": pd.Series([0.0], dtype="float64"), "width": pd.Series([0.0], dtype="float64"), "depth": pd.Series([0.0], dtype="float64"), "eq_width": pd.Series([0.0], dtype="float64"), "membrane": pd.Series(["example_value"], dtype="object"), "MWCO": pd.Series([0], dtype="int64"), "SR": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0.0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is the model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs/model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise
    
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return json.dumps({"error": result})
