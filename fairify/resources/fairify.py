import json
import numpy as np
from flask import jsonify, request
from flask_restful import Resource, reqparse

class BinaryLabelDatasetMetric(Resource):
    def demographic_parity(self, target, pred, group):
        return np.abs(np.mean(pred[group == 0]) - np.mean(pred[group==1]))

    def equalized_opportunity(self, target, pred, group):
        return np.abs(np.mean(pred[(group == 0)&target]) - np.mean(pred[(group==1)&target]))

    def acceptance_rate_parity(self, target, pred, group, percentile=80):
        threshold = np.percentile(pred, percentile)
        mask = group == 0
        return np.var([np.sum(pred[mask] > threshold) / np.sum(mask), np.sum(pred[~mask] > threshold) / np.sum(~mask)])

    def get_metrics_func(self, name):
        name_to_func = {"demographic_parity": self.demographic_parity, "equalized_opportunity": self.equalized_opportunity, "acceptance_rate_parity": self.acceptance_rate_parity}
        return name_to_func[name]

    def post(self):
        json_data = request.get_json(force=True)
        target = np.array(json_data["target"]).astype(int)
        pred = np.array(json_data["pred"])
        group = np.array(json_data["group"]).astype(int)
        if "metrics" in json_data:
            metrics = json_data["metrics"]
        else:            
            metrics = ["demographic_parity", "equalized_opportunity", "acceptance_rate_parity"]
        return jsonify({metric: self.get_metrics_func(metric)(target, pred, group) for metric in metrics})
        

class RecommendationMetric(Resource):
    def __init__(self, **kwargs):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("uid", type=str)

    def get(self):
        args = self.parser.parse_args()
        uid = self.user_uid_map[args["uid"]]

        res = {}
        return res