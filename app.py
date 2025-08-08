import os
from datetime import timedelta
import random
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image

# --- Setup paths and upload folder ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secret-key"  # Change this to a strong secret key
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.permanent_session_lifetime = timedelta(days=7)

db = SQLAlchemy(app)

# ---------- Models ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True, nullable=False)
    address = db.Column(db.String(300))
    occupation = db.Column(db.String(120))
    profile_photo = db.Column(db.String(300))
    password_hash = db.Column(db.String(300))

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ---------- Initialize database ----------
with app.app_context():
    db.create_all()

# ---------- Helper functions ----------
def login_user(user):
    session.permanent = True
    session["user_id"] = user.id
    session["user_name"] = user.name
    session["user_email"] = user.email
    session["user_photo"] = user.profile_photo

def logout_user():
    session.clear()

def current_user():
    uid = session.get("user_id")
    if uid:
        return User.query.get(uid)
    return None

# ---------- Disease class names ----------
class_names = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

# ---------- Disease solutions with medicine image ----------
disease_solutions = {
    "bacterial_leaf_blight": {
        "name": "Bacterial Leaf Blight",
        "medicine": "Copper-based bactericides (e.g., Copper oxychloride), Streptomycin, Kasugamycin",
        "control": "Use resistant varieties, proper field sanitation",
        "medicine_image": "Kocide.jpg"
    },
    "bacterial_leaf_streak": {
        "name": "Bacterial Leaf Streak",
        "medicine": "Copper fungicides (Copper oxychloride), Antibiotics like Streptomycin",
        "control": "Seed treatment, resistant varieties",
        "medicine_image": "Kocide.jpg"
    },
    "bacterial_panicle_blight": {
        "name": "Bacterial Panicle Blight",
        "medicine": "Copper-based bactericides, Streptomycin",
        "control": "Use disease-free seeds, crop rotation",
        "medicine_image": "Kocide.jpg"
    },
    "blast": {
        "name": "Blast (Rice blast fungus, Magnaporthe oryzae)",
        "medicine": "Fungicides such as Tricyclazole, Isoprothiolane, Carbendazim",
        "control": "Resistant varieties and proper nitrogen management",
        "medicine_image": "Tricyclazole 75 WP.jpg"
    },
    "brown_spot": {
        "name": "Brown Spot",
        "medicine": "Fungicides like Mancozeb, Copper oxychloride",
        "control": "Balanced fertilization, proper water management",
        "medicine_image": "Blitox 50 WP.jpg"
    },
    "dead_heart": {
        "name": "Dead Heart (Stem borer damage)",
        "medicine": "Insecticides such as Carbofuran, Chlorantraniliprole",
        "control": "Field sanitation, resistant varieties",
        "medicine_image": "Furadan 3G.jpg"
    },
    "downy_mildew": {
        "name": "Downy Mildew",
        "medicine": "Fungicides like Metalaxyl (Ridomil), Copper oxychloride",
        "control": "Use resistant varieties, good drainage",
        "medicine_image": "Ridomil Gold MZ.jpg"
    },
    "hispa": {
        "name": "Hispa (Rice hispa beetle)",
        "medicine": "Insecticides such as Quinalphos, Chlorpyrifos",
        "control": "Early planting, removal of weed hosts",
        "medicine_image": "Beam 75 WP.jpg"
    },
    "normal": {
        "name": "Normal",
        "medicine": "No disease; no medicine needed.",
        "control": "Maintain good agricultural practices",
        "medicine_image": None
    },
    "tungro": {
        "name": "Tungro (Rice tungro virus transmitted by leafhoppers)",
        "medicine": "No direct medicine; control vectors using insecticides such as Imidacloprid, Carbofuran",
        "control": "Use resistant varieties and proper field management",
        "medicine_image": "Confidor.jpg"
    },
}

# ---------- Nutrient levels, symptoms and recommendations ----------
nutrient_levels = {
    "nitrogen": {
        "low": 50,
        "optimal_min": 50,
        "optimal_max": 100,
        "high": 150,
        "symptoms_low": [
            "Yellowing older leaves (chlorosis)",
            "Stunted growth",
            "Poor tillering"
        ],
        "fertilizer_low": "Apply 80–120 kg N/ha (Urea, Ammonium sulfate) in split doses (transplanting, tillering, panicle)",
        "symptoms_high": [
            "Excessive vegetative growth",
            "Delayed maturity",
            "Lodging risk",
            "Poor grain filling"
        ],
        "solution_high": [
            "Reduce nitrogen fertilizer",
            "Balanced fertilization",
            "Avoid excessive irrigation"
        ]
    },
    "phosphorus": {
        "low": 10,
        "optimal_min": 10,
        "optimal_max": 30,
        "high": 50,
        "symptoms_low": [
            "Dark green leaves but poor tillering",
            "Stunted root growth",
            "Delayed maturity"
        ],
        "fertilizer_low": "Apply 30–60 kg P2O5/ha (SSP, TSP, DAP) at transplanting for best uptake",
        "symptoms_high": [
            "Micronutrient (Zn, Fe) deficiencies due to antagonism"
        ],
        "solution_high": [
            "Avoid excessive P application",
            "Soil test before fertilization"
        ]
    },
    "potassium": {
        "low": 80,
        "optimal_min": 80,
        "optimal_max": 200,
        "high": 250,
        "symptoms_low": [
            "Yellowing and drying leaf edges (marginal scorch)",
            "Weak stems and lodging",
            "Poor grain filling and quality"
        ],
        "fertilizer_low": "Apply 40–80 kg K2O/ha (MOP, SOP) split application (transplanting, tillering)",
        "symptoms_high": [],
        "solution_high": []
    }
}

def analyze_nutrient_level(nutrient, value):
    info = nutrient_levels.get(nutrient)
    if not info:
        return None

    if value < info["low"]:
        status = "Low"
        symptoms = info["symptoms_low"]
        recommendation = info["fertilizer_low"]
        issues_high = []
        solution_high = []
    elif info["optimal_min"] <= value <= info["optimal_max"]:
        status = "Optimal"
        symptoms = []
        recommendation = "Nutrient level is optimal."
        issues_high = []
        solution_high = []
    elif value > info["high"]:
        status = "High"
        symptoms = info.get("symptoms_high", [])
        recommendation = ""
        issues_high = symptoms
        solution_high = info.get("solution_high", [])
    else:
        # Between optimal_max and high but not considered high
        status = "Slightly High"
        symptoms = []
        recommendation = "Be cautious of potential nutrient imbalance."
        issues_high = []
        solution_high = []

    return {
        "status": status,
        "symptoms_if_low": info.get("symptoms_low", []),
        "fertilizer_if_low": info.get("fertilizer_low", ""),
        "symptoms_if_high": symptoms,
        "solution_if_high": solution_high,
        "recommendation": recommendation
    }

# ---------- Load PyTorch Model once ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(BASE_DIR, 'best_efficientnet_b4.pth')  # Put your model here
image_size = 380

weights = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
auto_transforms = weights.transforms()
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=auto_transforms.mean, std=auto_transforms.std),
])

model = torchvision.models.efficientnet_b4(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_rice_disease(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item() * 100

# ---------- Routes ----------

@app.route("/")
def index():
    return render_template("home.html", user=current_user())

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        address = request.form.get("address", "").strip()
        occupation = request.form.get("occupation", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not all([name, email, password, confirm]):
            flash("Please fill required fields", "error")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("login"))

        file = request.files.get("profile_photo")
        filename = None
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        new_user = User(
            name=name,
            email=email,
            address=address,
            occupation=occupation,
            profile_photo=filename,
            password_hash=generate_password_hash(password),
        )
        db.session.add(new_user)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            flash("An error occurred during registration. Please try again.", "error")
            return redirect(url_for("register"))

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", user=current_user())

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("No account found with that email. Please register.", "error")
            return redirect(url_for("register"))
        if not user.check_password(password):
            flash("Password incorrect. Try again.", "error")
            return redirect(url_for("login"))

        login_user(user)
        flash("Logged in successfully.", "success")
        return redirect(url_for("index"))
    return render_template("login.html", user=current_user())

@app.route("/logout")
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

@app.route("/profile")
def profile():
    user = current_user()
    if not user:
        flash("Please login to view profile.", "error")
        return redirect(url_for("login"))
    return render_template("profile.html", user=user)

@app.route("/rice-disease", methods=["GET", "POST"])
def rice_disease():
    prediction = None
    confidence = None
    solution = None
    filename = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an image.", "error")
            return redirect(url_for("rice_disease"))
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            prediction, confidence = predict_rice_disease(save_path)
            solution = disease_solutions.get(prediction)
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
            return redirect(url_for("rice_disease"))

    return render_template(
        "rice_disease.html",
        user=current_user(),
        prediction=prediction,
        confidence=confidence,
        solution=solution,
        filename=filename,
    )

@app.route("/soil-test")
def soil_test():
    return render_template("soil_test.html", user=current_user())

@app.route("/soil-data")
def soil_data():
    data = {
        "nitrogen": round(random.uniform(0, 50), 2),
        "phosphorus": round(random.uniform(0, 50), 2),
        "potassium": round(random.uniform(0, 50), 2),
    }
    return jsonify(data)

# Modified here to render soil_text.html instead of soil_report.html
@app.route("/soil-report", methods=["GET", "POST"])
def soil_report():
    report = {}
    nitrogen = phosphorus = potassium = None
    error = None

    if request.method == "POST":
        try:
            nitrogen = float(request.form.get("nitrogen"))
            phosphorus = float(request.form.get("phosphorus"))
            potassium = float(request.form.get("potassium"))
        except (TypeError, ValueError):
            error = "Please enter valid numeric values for all nutrients."

        if not error:
            report["nitrogen"] = analyze_nutrient_level("nitrogen", nitrogen)
            report["phosphorus"] = analyze_nutrient_level("phosphorus", phosphorus)
            report["potassium"] = analyze_nutrient_level("potassium", potassium)

    return render_template(
        "soil_text.html",  # <-- changed here
        user=current_user(),
        report=report,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        error=error
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
