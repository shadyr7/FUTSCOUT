# ⚽ FUTSCOUT: Position-Aware Deep Learning System for Intelligent Football Player Scouting

**Live Demo:** [FUTSCOUT Web App](https://futscout-position-aware-deep-learning.onrender.com)

FUTSCOUT is an intelligent football analytics platform designed to go beyond basic statistics and deliver **position-aware, context-sensitive performance ratings** using deep learning. It integrates **Expected Goals (xG)**, **Expected Assists (xA)**, and advanced per-90 statistics to generate automated scouting reports through an intuitive web-based interface.

---

## 🚀 Key Features

- 🎯 **Position-Aware Attention Neural Network (PAANN):** Custom deep learning model that dynamically prioritizes features based on player position.
- 🌟 **xA Prediction:** Estimated using a trained Random Forest model based on passing and creative playmaking stats.
- 📊 **Advanced Metrics:**
  - xG, xA, G–xG, A–xA
  - Per-90 statistics for fair cross-player comparisons
- 📈 **Smart Visuals:**
  - Radar charts for skill profiles
  - Position heatmaps
  - Automated verdict on performance tier
- 🌐 **Clean Web Interface:**
  - Built using Flask
  - Interactive scouting report generator

---

## 🧠 Model Architecture

**PAANN – Position-Aware Attention Neural Network**

- **Position Embedding:** Encodes player role into the model
- **Feature + Position Concatenation:** Adds positional context to raw features
- **Attention Mechanism:** Weighs features based on role-specific importance
- **Encoder:** Dense layers with BatchNorm + Dropout for stable learning
- **Output:** Predicts real-valued player rating (scale of 0–10)

---

## 📁 Tech Stack

| Component        | Technology             |
|------------------|-------------------------|
| Model Training   | PyTorch                 |
| xA Prediction    | Random Forest (Sklearn) |
| Frontend + API   | Flask                   |
| Visualization    | matplotlib, seaborn, mplsoccer |
| Deployment       | Render                  |

---

## 🛠️ How It Works

1. Enter player performance stats (attacking, passing, defensive)
2. The PAANN model predicts the player's role-aware rating
3. xA is computed using a trained Random Forest model
4. A scouting report is generated with:
   - Rating
   - Radar chart
   - Advanced stats (xG, xA, G–xG, A–xA)
   - Verdict on player performance

---

## 📊 Model Performance

| Metric        | Training Set | Validation Set |
|---------------|--------------|----------------|
| MSE           | 0.0040       | 0.0065         |
| MAE           | 0.0431       | 0.0504         |
| R² Score      | 0.9727       | 0.9568         |
| Accuracy      | 94.23%       | 92.86%         |
| F1 Score      | 91.52%       | 85.04%         |

---

## ⚠️ Limitations

- No match-by-match or time-series analysis (season-aggregated only)
- Goalkeepers not included
- Approximate xG due to lack of granular shot data
- Single-head attention (multi-head attention in future plans)

---

## 🔮 Future Enhancements

- Live match feed integration
- Mobile and API version
- Player comparison tools
- Graph Neural Networks and more advanced DL models
- Expanded dataset and real-time scouting

---
