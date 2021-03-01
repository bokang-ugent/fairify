import json
import os

import numpy as np
import pickle
from flask import flash, jsonify, request, redirect, url_for
from flask_restful import Resource, reqparse
from werkzeug.utils import secure_filename

from .metrics import compute_representation_bias, compute_fairness_metrics

class BinaryLabelDatasetMetric(Resource):
    def __init__(self, **kwargs):
        self.ALLOWED_EXTENSIONS = kwargs.get("ALLOWED_EXTENSIONS")

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

class RepresentationMetric(Resource):
    def __init__(self, **kwargs):
        self.ALLOWED_EXTENSIONS = kwargs.get("ALLOWED_EXTENSIONS")
        self.UPLOAD_FOLDER = kwargs.get("UPLOAD_FOLDER")
        if not os.path.exists(self.UPLOAD_FOLDER):
            os.makedirs(self.UPLOAD_FOLDER)

    def allowed_file(self, filename, ALLOWED_EXTENSIONS):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def post(self):
        # check if the post request has the file part        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename                
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and self.allowed_file(file.filename, self.ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.UPLOAD_FOLDER, filename)
            file.save(filepath)
            data = pickle.load(open(filepath, "rb"))
            res = compute_representation_bias(data["user_emb"], data["group"])
            return res
        return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
            </form>
            '''

class RecommendationMetric(Resource):
    def __init__(self, **kwargs):
        self.ALLOWED_EXTENSIONS = kwargs.get("ALLOWED_EXTENSIONS")
        self.UPLOAD_FOLDER = kwargs.get("UPLOAD_FOLDER")
        if not os.path.exists(self.UPLOAD_FOLDER):
            os.makedirs(self.UPLOAD_FOLDER)

    def allowed_file(self, filename, ALLOWED_EXTENSIONS):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def post(self):
        # check if the post request has the file part        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename                
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and self.allowed_file(file.filename, self.ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.UPLOAD_FOLDER, filename)
            file.save(filepath)
            data = pickle.load(open(filepath, "rb"))
            res = compute_fairness_metrics(data["A"], data["E"], data["label"], data["model"], data["uid_feature_map"])
            return res
        return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
            </form>
            '''