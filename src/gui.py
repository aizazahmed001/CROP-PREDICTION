"""
gui.py
------
Tkinter desktop GUI for the Smart Agriculture Decision Support System.

Features:
  • Input fields for all 7 soil/climate features
  • Three prediction outputs: Crop, Soil Zone, Yield
  • Embedded Matplotlib plots (Feature Importance, Cluster, Residuals)
  • Predict / Clear buttons
  • Status bar and tabbed visualization panel
"""

import os
import sys
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
# sys.path insertion is not required because the GUI uses local model files directly

# ── Colour Palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#F4F7F0",
    "panel":     "#FFFFFF",
    "accent":    "#4A7C59",
    "accent2":   "#2D6A4F",
    "btn":       "#52B788",
    "btn_hover": "#40916C",
    "btn_clear": "#D62828",
    "text":      "#1B1B1B",
    "subtext":   "#555555",
    "border":    "#C8D8C8",
    "header_bg": "#2D6A4F",
    "header_fg": "#FFFFFF",
    "result_bg": "#EAF4EE",
}


# ── Model Loader ──────────────────────────────────────────────────────────────
def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Please run 'python train.py' first to train and save all models."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main Application ──────────────────────────────────────────────────────────
class SmartAgriApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌾 Smart Agriculture Decision Support System")
        self.geometry("1100x720")
        self.resizable(True, True)
        self.configure(bg=COLORS["bg"])

        # Load models
        try:
            self.dt     = load_model("decision_tree")
            self.km     = load_model("knn")
            self.lr     = load_model("linear_regression")
            self.le     = load_model("label_encoder")
            self.scaler = load_model("scaler")
            self.pca    = load_model("pca")
        except FileNotFoundError as e:
            messagebox.showerror("Models Not Found", str(e))
            self.destroy()
            return

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()

        main_frame = tk.Frame(self, bg=COLORS["bg"])
        main_frame.pack(fill="both", expand=True, padx=12, pady=8)

        # Left: inputs + results
        left = tk.Frame(main_frame, bg=COLORS["bg"], width=360)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        self._build_input_panel(left)
        self._build_result_panel(left)

        # Right: visualization notebook
        right = tk.Frame(main_frame, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._build_viz_panel(right)

        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=COLORS["header_bg"], height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr,
                 text="🌾  Smart Agriculture Decision Support System",
                 font=("Segoe UI", 16, "bold"),
                 bg=COLORS["header_bg"], fg=COLORS["header_fg"]
                 ).pack(side="left", padx=18, pady=12)

        tk.Label(hdr,
                 text="AI-Powered  |  Decision Tree · KMeans · Linear Regression",
                 font=("Segoe UI", 9),
                 bg=COLORS["header_bg"], fg="#A8D8B9"
                 ).pack(side="right", padx=18)

    def _build_input_panel(self, parent):
        panel = tk.LabelFrame(parent,
                              text=" 🧪 Soil & Climate Inputs ",
                              font=("Segoe UI", 10, "bold"),
                              bg=COLORS["panel"], fg=COLORS["accent2"],
                              bd=1, relief="groove",
                              padx=12, pady=8)
        panel.pack(fill="x", pady=(0, 8))

        fields = [
            ("Nitrogen (N, mg/kg)",   "nitrogen",    "90"),
            ("Phosphorus (P, mg/kg)", "phosphorus",  "42"),
            ("Potassium (K, mg/kg)",  "potassium",   "43"),
            ("Temperature (°C)",      "temperature", "25.0"),
            ("Humidity (%)",          "humidity",    "80.0"),
            ("Soil pH",               "ph",          "6.5"),
            ("Rainfall (mm)",         "rainfall",    "200.0"),
        ]

        self.entries = {}
        for label_text, key, default in fields:
            row = tk.Frame(panel, bg=COLORS["panel"])
            row.pack(fill="x", pady=2)

            tk.Label(row, text=label_text, font=("Segoe UI", 9),
                     bg=COLORS["panel"], fg=COLORS["text"],
                     width=22, anchor="w"
                     ).pack(side="left")

            var = tk.StringVar(value=default)
            entry = tk.Entry(row, textvariable=var, font=("Segoe UI", 9),
                             width=12, relief="flat",
                             bg="#F0F4F0", fg=COLORS["text"],
                             highlightthickness=1,
                             highlightbackground=COLORS["border"],
                             highlightcolor=COLORS["accent"])
            entry.pack(side="left", padx=(4, 0))
            self.entries[key] = var

        # Buttons
        btn_row = tk.Frame(panel, bg=COLORS["panel"])
        btn_row.pack(fill="x", pady=(10, 2))

        predict_btn = tk.Button(btn_row, text="🔍  PREDICT",
                                command=self._predict,
                                font=("Segoe UI", 10, "bold"),
                                bg=COLORS["btn"], fg="white",
                                activebackground=COLORS["btn_hover"],
                                relief="flat", cursor="hand2",
                                padx=18, pady=6)
        predict_btn.pack(side="left", padx=(0, 8))

        clear_btn = tk.Button(btn_row, text="✕  CLEAR",
                              command=self._clear,
                              font=("Segoe UI", 10),
                              bg=COLORS["btn_clear"], fg="white",
                              activebackground="#9B1C1C",
                              relief="flat", cursor="hand2",
                              padx=18, pady=6)
        clear_btn.pack(side="left")

    def _build_result_panel(self, parent):
        panel = tk.LabelFrame(parent,
                              text=" 📊 Prediction Results ",
                              font=("Segoe UI", 10, "bold"),
                              bg=COLORS["panel"], fg=COLORS["accent2"],
                              bd=1, relief="groove",
                              padx=12, pady=8)
        panel.pack(fill="x")

        results = [
            ("🌱 Recommended Crop",   "crop_result"),
            ("🗺️ Soil Zone (Cluster)", "cluster_result"),
            ("📦 Est. Yield (kg/ha)",  "yield_result"),
        ]

        self.result_vars = {}
        for label_text, key in results:
            tk.Label(panel, text=label_text,
                     font=("Segoe UI", 9, "bold"),
                     bg=COLORS["panel"], fg=COLORS["subtext"]
                     ).pack(anchor="w")

            var = tk.StringVar(value="—")
            lbl = tk.Label(panel, textvariable=var,
                           font=("Segoe UI", 13, "bold"),
                           bg=COLORS["result_bg"], fg=COLORS["accent2"],
                           relief="flat", width=28, anchor="center",
                           pady=4)
            lbl.pack(fill="x", pady=(0, 8))
            self.result_vars[key] = var

    def _build_viz_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Segoe UI", 9, "bold"), padding=[10, 4])

        self.fig_frames = {}
        tabs = [
            ("Feature Importance", "fi_tab"),
            ("Soil Clusters",       "cl_tab"),
            ("Yield Residuals",     "res_tab"),
        ]
        for tab_title, key in tabs:
            frame = tk.Frame(nb, bg=COLORS["panel"])
            nb.add(frame, text=f"  {tab_title}  ")
            self.fig_frames[key] = frame

        # Load saved plots into tabs
        self._embed_saved_plot("feature_importance.png", "fi_tab")
        self._embed_saved_plot("cluster_plot.png",       "cl_tab")
        self._embed_saved_plot("residual_plot.png",      "res_tab")

    def _embed_saved_plot(self, filename, tab_key):
        """Load a PNG result and display it inside the notebook tab."""
        path  = os.path.join(RESULTS_DIR, filename)
        frame = self.fig_frames[tab_key]

        if not os.path.exists(path):
            tk.Label(frame,
                     text=f"⚠️  {filename} not found.\nRun train.py first.",
                     font=("Segoe UI", 10), bg=COLORS["panel"],
                     fg="#AA0000"
                     ).pack(expand=True)
            return

        # Use Matplotlib to re-render the image (avoids PIL resize artefacts)
        fig = Figure(figsize=(6.5, 4.5), dpi=100)
        ax  = fig.add_subplot(111)
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout(pad=0)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="  Ready – enter values and press Predict")
        bar = tk.Label(self, textvariable=self.status_var,
                       font=("Segoe UI", 8), bd=1, relief="sunken",
                       anchor="w", bg="#D8EDD8", fg=COLORS["subtext"], padx=8)
        bar.pack(fill="x", side="bottom")

    # ── Logic ─────────────────────────────────────────────────────────────────
    def _get_inputs(self):
        keys = ["nitrogen", "phosphorus", "potassium",
                "temperature", "humidity", "ph", "rainfall"]
        values = []
        for k in keys:
            raw = self.entries[k].get().strip()
            if not raw:
                raise ValueError(f"Field '{k}' is empty.")
            values.append(float(raw))
        return np.array(values).reshape(1, -1)

    def _predict(self):
        try:
            X_raw = self._get_inputs()
        except ValueError as e:
            messagebox.showwarning("Input Error", str(e))
            return

        # Scale
        X_scaled = self.scaler.transform(X_raw)

        # 1. Crop recommendation
        crop_enc  = self.dt.predict(X_scaled)[0]
        crop_name = self.le.inverse_transform([crop_enc])[0].capitalize()
        self.result_vars["crop_result"].set(crop_name)

        # 2. Soil cluster
        cluster   = self.km.predict(X_scaled)[0]
        self.result_vars["cluster_result"].set(f"Zone {cluster + 1}")

        # 3. Yield prediction
        yield_val = self.lr.predict(X_scaled)[0]
        self.result_vars["yield_result"].set(f"{yield_val:,.1f} kg/ha")

        self.status_var.set(
            f"  ✅ Prediction complete – Crop: {crop_name} | "
            f"Zone: {cluster + 1} | Yield: {yield_val:,.1f} kg/ha"
        )

    def _clear(self):
        defaults = {
            "nitrogen": "90", "phosphorus": "42", "potassium": "43",
            "temperature": "25.0", "humidity": "80.0",
            "ph": "6.5", "rainfall": "200.0",
        }
        for k, v in defaults.items():
            self.entries[k].set(v)
        for var in self.result_vars.values():
            var.set("—")
        self.status_var.set("  Ready – enter values and press Predict")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SmartAgriApp()
    app.mainloop()
