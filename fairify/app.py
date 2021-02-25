import os

from flask import Flask
from flask_restful import Api


from resources.fairify import BinaryLabelDatasetMetric


BINARY_LABEL_METRIC_API = os.environ["BINARY_LABEL_METRIC_API"]

app = Flask(__name__)
api = Api(app)


kwargs = {}
api.add_resource(BinaryLabelDatasetMetric, BINARY_LABEL_METRIC_API, resource_class_kwargs=kwargs)