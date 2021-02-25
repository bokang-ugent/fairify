import json
import numpy as np
from flask_restful import Resource, reqparse


class Fairify(Resource):
    def __init__(self, **kwargs):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("uid", type=str)

    def get(self):
        args = self.parser.parse_args()
        uid = self.user_uid_map[args["uid"]]

        res = {}
        return res