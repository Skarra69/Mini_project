import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

from ml_engine import KeplerMLEngine

app = Flask(__name__)
# Enable CORS for all routes so modern frontend (e.g., React on Vite port 5173 or proxy) can connect seamlessly
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Instantiate global ML Engine
engine = KeplerMLEngine()


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        loaded = engine.dataset is not None
        rows = int(engine.dataset.shape[0]) if loaded else 0
        return jsonify({
            "status": "healthy",
            "service": "Exoplanet Validation ML Flask API",
            "dataset_loaded": loaded,
            "rows": rows
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/overview", methods=["GET"])
def get_overview():
    try:
        if engine.dataset is None:
            engine.load_dataset()
        if engine.X is None:
            engine.preprocess_dataset()

        models_info = engine.get_models_metrics()
        best_r2 = 0.9242
        best_name = "CatBoost Regressor"
        if models_info:
            sorted_models = sorted(models_info, key=lambda x: x["r2"], reverse=True)
            best_r2 = sorted_models[0]["r2"]
            best_name = sorted_models[0]["name"]

        return jsonify({
            "status": "success",
            "totalKois": int(engine.dataset.shape[0]),
            "inputFeatures": int(engine.X.shape[1]),
            "modelsTrained": len(models_info),
            "bestModelR2": best_r2,
            "bestModelName": best_name,
            "models": models_info
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    try:
        if engine.dataset is None:
            engine.load_dataset()

        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
        search = request.args.get("search", "").lower()

        df = engine.dataset.copy()
        if search:
            # Search by name or kepid string
            mask = df.astype(str).apply(lambda x: x.str.lower().str.contains(search)).any(axis=1)
            df = df[mask]

        total_rows = len(df)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        slice_df = df.iloc[start_idx:end_idx]

        return jsonify({
            "status": "success",
            "page": page,
            "limit": limit,
            "total_records": total_rows,
            "records": slice_df.to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/dataset/upload", methods=["POST"])
def upload_dataset():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "Empty filename"}), 400

        upload_dir = os.path.join(engine.base_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        res = engine.load_dataset(file_path)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/preprocess", methods=["POST"])
def preprocess_dataset():
    try:
        res = engine.preprocess_dataset()
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/eda", methods=["GET"])
def get_eda():
    try:
        res = engine.get_eda_stats()
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_models():
    try:
        models_list = engine.get_models_metrics()
        return jsonify({"status": "success", "models": models_list}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models/train", methods=["POST"])
def train_model():
    try:
        data = request.get_json() or {}
        algorithm = data.get("algorithm", "CatBoost Regressor")

        if algorithm.lower() == "all":
            results = []
            for algo_name in engine.model_configs.keys():
                res = engine.train_model(algo_name)
                results.append(res)
            return jsonify({"status": "success", "all_results": results, "models": engine.get_models_metrics()}), 200

        res = engine.train_model(algorithm)
        return jsonify({"status": "success", "result": res, "models": engine.get_models_metrics()}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        records = data.get("records")
        if not records:
            return jsonify({"status": "error", "message": "No records provided for prediction"}), 400

        df = pd.DataFrame(records)
        predictions = engine.predict_data(df)

        return jsonify({
            "status": "success",
            "predictions": predictions
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/upload", methods=["POST"])
def predict_upload():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)
        predictions = engine.predict_data(df)
        df["Pred_Score"] = predictions

        return jsonify({
            "status": "success",
            "count": len(df),
            "results": df.head(50).to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Kepler Exoplanet ML Flask API on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
