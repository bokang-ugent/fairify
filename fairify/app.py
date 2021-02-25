import os

from flask import Flask
from flask_restful import Api


from resources.fairify import Fairify


API_URI = os.environ["API_URI"]

app = Flask(__name__)
api = Api(app)


kwargs = {}
api.add_resource(Fairify, API_URI, resource_class_kwargs=kwargs)