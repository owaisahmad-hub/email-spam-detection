# spam_gui.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
import joblib

# ML imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from datetime import datetime


MODEL_FILE = "spam_model.joblib"
VECTORIZER_FILE = "vectorizer.joblib"
DATA_FILE = "spam.csv"   # Put your CSV (with columns v1=label, v2=text) here

#  ML 
def ensure_model():
    """
    Train the model on first run if .joblib files are missing.
    Then load and return (model, vectorizer).
    """
    if not (os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE)):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"Could not find '{DATA_FILE}'. Put your dataset in this folder or change DATA_FILE."
            )

        # Load and prepare data
        df = pd.read_csv(DATA_FILE, encoding="latin-1")
        # Expect standard SMS Spam Collection format
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        df = df.dropna(subset=["label", "text"])

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LinearSVC()
        model.fit(X_train_vec, y_train)

        # Save for reuse
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)

    # Load trained artifacts
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer

# Prediction helper
def predict_label(text):
    vec = VECTORIZER.transform([text])
    pred = MODEL.predict(vec)[0]
    return "SPAM", "❌" if pred == 1 else ("HAM", "✅")

#  Tkinter UI 
def build_ui(root):
    root.title("📧 Spam Detector — Step 1")
    root.geometry("720x520")
    root.minsize(680, 480)
    root.configure(bg="#f5f6fa")

    # Make it responsive
    root.grid_rowconfigure(2, weight=1)   # main area grows
    root.grid_columnconfigure(0, weight=1)

    # ttk styling
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except:
        pass

    style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), background="#f5f6fa")
    style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#6b7280", background="#f5f6fa")
    style.configure("TButton", font=("Segoe UI", 11), padding=8)
    style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"))
    style.map("TButton", relief=[("pressed", "sunken"), ("active", "raised")])

    # Header
    header = ttk.Frame(root, padding=(16, 14))
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    title = ttk.Label(header, text="Spam Detector (Step 1)", style="Title.TLabel")
    subtitle = ttk.Label(header, text="Type a message and check if it’s spam or ham.", style="Sub.TLabel")
    title.grid(row=0, column=0, sticky="w")
    subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))

    # Input area
    input_card = tk.Frame(root, bg="white", bd=0, highlightthickness=0)
    input_card.grid(row=1, column=0, sticky="ew", padx=16, pady=(6, 8))
    input_card.grid_columnconfigure(0, weight=1)

    lbl = ttk.Label(input_card, text="Message:", font=("Segoe UI", 11))
    lbl.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))

    text_frame = tk.Frame(input_card, bg="white")
    text_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
    text_frame.grid_columnconfigure(0, weight=1)

    text_box = tk.Text(text_frame, height=6, wrap="word", font=("Segoe UI", 11), bd=0)
    text_box.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(text_frame, command=text_box.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    text_box.configure(yscrollcommand=scroll.set)

    # Buttons
    btns = ttk.Frame(root, padding=(16, 0))
    btns.grid(row=2, column=0, sticky="ew")
    btns.grid_columnconfigure(0, weight=1)
    btns.grid_columnconfigure(1, weight=0)
    btns.grid_columnconfigure(2, weight=0)

    def on_check():
        msg = text_box.get("1.0", "end").strip()
        if not msg:
            messagebox.showwarning("Input needed", "Please type a message to check.")
            return
        label, emoji = predict_label(msg)
        set_result(label, emoji)

    def on_clear():
        text_box.delete("1.0", "end")
        set_result("", "")

    check_btn = ttk.Button(btns, text="Check Spam", style="Accent.TButton", command=on_check)
    clear_btn = ttk.Button(btns, text="Clear", command=on_clear)
    check_btn.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=8)
    clear_btn.grid(row=0, column=2, sticky="e", pady=8)

    # Result card
    result_card = tk.Frame(root, bg="white")
    result_card.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))
    root.grid_rowconfigure(3, weight=1)
    result_card.grid_columnconfigure(0, weight=1)
    result_card.grid_rowconfigure(0, weight=1)

    # Big result badge
    result_var = tk.StringVar(value="")
    emoji_var = tk.StringVar(value="")

    badge = tk.Label(
        result_card,
        textvariable=emoji_var,
        font=("Segoe UI Emoji", 40),
        bg="white",
    )
    badge.place(relx=0.5, rely=0.32, anchor="center")

    res_label = tk.Label(
        result_card,
        textvariable=result_var,
        font=("Segoe UI", 18, "bold"),
        bg="white",
    )
    res_label.place(relx=0.5, rely=0.6, anchor="center")

    # helper to set result
    def set_result(label, emoji):
        emoji_var.set(emoji)
        if label == "SPAM":
            res_label.config(text="SPAM", fg="#dc2626")  # red
        elif label == "HAM":
            res_label.config(text="HAM", fg="#16a34a")  # green
        else:
            res_label.config(text="", fg="#111827")

    # Footer hint
    footer = ttk.Label(
        root,
        text="Tip: First run may take a moment to train from spam.csv; next runs are instant.",
        style="Sub.TLabel",
        anchor="center",
    )
    footer.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 10))

# -------------------- Main --------------------
if __name__ == "__main__":
    try:
        MODEL, VECTORIZER = ensure_model()
    except Exception as e:
        messagebox.showerror("Model error", str(e))
        raise

    app = tk.Tk()
    build_ui(app)
    app.mainloop()
