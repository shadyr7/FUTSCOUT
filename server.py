import torch
import torch.nn as nn
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from model_architecture import PositionAwareAttentionModel  # Ensure this is defined properly
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# === 🔧 Load saved components in correct order ===

scaler = joblib.load("feature_scaler.pkl")
position_to_idx = joblib.load("position_to_idx.pkl")
model = PositionAwareAttentionModel(input_size=16, num_positions=len(position_to_idx))
model.load_state_dict(torch.load("rating_model.pth", map_location=torch.device('cpu')))
model.eval()
xa_model = joblib.load("xA_random_forest_model.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

basic_input_features = [
    'mins', 'goals', 'asists', 'shots_taken', 'crosses', 'longb', 'thrb',
    'keyp', 'avgp', 'drb', 'spg', 'tackles', 'owng', 'clear', 'inter', 'blocks'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/report")
def report():
    return render_template("report.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        stats = [float(data[feat]) for feat in basic_input_features]
        position = data["position"]

        stats_scaled = scaler.transform([stats])
        x = torch.tensor(stats_scaled, dtype=torch.float32).to(device)
        pos_idx = torch.tensor([position_to_idx[position]], dtype=torch.long).to(device)

        with torch.no_grad():
            rating = model(x, pos_idx).item()

        mins = float(data['mins'])
        crosses_p90 = float(data['crosses']) / (mins / 90) if mins > 0 else 0
        xa_features = np.array([[float(data['keyp']), float(data['thrb']), float(data['longb']), float(data['avgp']), crosses_p90]])
        xA = xa_model.predict(xa_features)[0]
        xApg = xA / (mins / 90) if mins > 0 else 0

        spg = float(data['spg'])
        shots_taken = spg * (mins / 90) if mins > 0 else 0
        expected_conversion_rate = 0.15 if position in ['ST', 'CF', 'LW', 'RW'] else 0.08
        goals = float(data['goals'])
        conversion_rate = (0.6 * (goals / shots_taken) + 0.4 * expected_conversion_rate) if shots_taken > 0 else expected_conversion_rate
        xG = shots_taken * conversion_rate
        xGpg = xG / (mins / 90) if mins > 0 else 0

        Passes90 = float(data['avgp']) * (mins / 90)
        KP90 = float(data['keyp']) * (mins / 90)
        Drb90 = float(data['drb']) * (mins / 90)

        G_xG = goals - xG
        A_xA = float(data['asists']) - xA

        scouting_data = {
            "player_name": data['name'],
            "team_name": data['team_name'],
            "league": data['league'],
            "jersey_no": data['jersey_no'],
            "height": data['height'],
            "position": position,
            "raw_stats": {feat: data[feat] for feat in basic_input_features},
            "advanced_stats": {
                "xA": round(xA, 2),
                "xApg": round(xApg, 2),
                "xG": round(xG, 2),
                "xGpg": round(xGpg, 2),
                "Passes/90": round(Passes90, 2),
                "KP/90": round(KP90, 2),
                "Drb/90": round(Drb90, 2),
                "crosses_p90": round(crosses_p90, 2),
                "conversion_rate": round(conversion_rate, 3),
                "expected_conversion_rate": round(expected_conversion_rate, 3),
                "G-xG": round(G_xG, 2),
                "A-xA": round(A_xA, 2),
            },
            "predicted_rating": round(rating, 2)
        }

        return jsonify(scouting_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
