from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from app.routes.auth_routes import auth
from app.routes.user_routes import users
from app.api.api import api
from app.routes.chart_routes import chart
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from app.routes.scrape import scrape_bp
import os

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = 'rahasia-capstone-therapalsy'
app.config['MONGO_URI'] = os.getenv("MONGO_URI")
mongo = PyMongo(app)
jwt = JWTManager(app)

# Register Blueprints
app.register_blueprint(auth, url_prefix='/api/auth')
app.register_blueprint(users, url_prefix='/api/user')
app.register_blueprint(api, url_prefix='/api')
app.register_blueprint(chart, url_prefix="/api/chart")
app.register_blueprint(scrape_bp, url_prefix='/api/scrape')
app.register_blueprint(scrape_bp, url_prefix='/api/detection')

@app.route('/')
def index():
    return "API Running"

print("[DEBUG] GOOGLE_CLIENT_ID:", os.getenv("GOOGLE_CLIENT_ID"))
