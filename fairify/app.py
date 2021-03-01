import os

from flask import Flask
from flask_restful import Api


from resources.fairify import BinaryLabelDatasetMetric, RepresentationMetric, RecommendationMetric


BINARY_LABEL_METRIC_API = os.environ["BINARY_LABEL_METRIC_API"]
REPRESENTATION_METRIC_API = os.environ["REPRESENTATION_METRIC_API"]
RECOMMENDATION_METRIC_API = os.environ["RECOMMENDATION_METRIC_API"]


app = Flask(__name__)
api = Api(app)
app.secret_key = "super secret key"
# app.config["SESSION_TYPE"] = "filesystem"

kwargs = {"ALLOWED_EXTENSIONS": os.environ["ALLOWED_EXTENSIONS"], "UPLOAD_FOLDER": os.environ["UPLOAD_FOLDER"]}
api.add_resource(BinaryLabelDatasetMetric, BINARY_LABEL_METRIC_API, resource_class_kwargs=kwargs)
api.add_resource(RepresentationMetric, REPRESENTATION_METRIC_API, resource_class_kwargs=kwargs)
api.add_resource(RecommendationMetric, RECOMMENDATION_METRIC_API, resource_class_kwargs=kwargs)