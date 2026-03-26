import os
import joblib

import numpy as np
import MySQLdb.cursors
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mail import Mail, Message
from flask_mysqldb import MySQL

# Load secrets from .env (ignored by Git)
load_dotenv()

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-dev-key-change-in-prod')

# Mail
app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 465
app.config['MAIL_USE_SSL']  = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(app)

# MySQL
app.config['MYSQL_HOST']     = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER']     = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB']       = os.getenv('MYSQL_DB', 'stroke_predictor')
mysql = MySQL(app)

# ── Model + threshold ──────────────────────────────────────────────────────────
MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
model     = joblib.load(os.path.join(MODELS_PATH, 'model.pkl'))
threshold = joblib.load(os.path.join(MODELS_PATH, 'threshold.pkl'))

# ── Preprocessing (shared with training notebooks) ─────────────────────────────
from utils.preprocessing import preprocess_input

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('homepage.html')



@app.route('/predict')
def predict():
    return render_template('home.html')

@app.route('/fast')
def fast():
    return render_template('fast.html')


@app.route('/manage')
def manage():
    return render_template('manage.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not name or not email or not password:
            message = 'Please fill out the form.'
        elif password != confirm:
            message = 'Passwords do not match.'
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM user WHERE email = %s', (email,))
            if cursor.fetchone():
                message = 'An account with that email already exists.'
            else:
                cursor.execute(
                    'INSERT INTO user (name, email, password) VALUES (%s, %s, %s)',
                    (name, email, password),
                )
                mysql.connection.commit()
                message = 'Registration successful! You can now log in.'
    return render_template('register.html', message=message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        cursor   = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM user WHERE email = %s AND password = %s',
            (email, password),
        )
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid']   = user['userid']
            session['name']     = user['name']
            session['email']    = user['email']
            return redirect(url_for('index'))
        else:
            message = 'Incorrect email or password.'
    return render_template('login.html', message=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/result', methods=['POST'])
def result():
    try:
        x = preprocess_input(
            gender            = request.form.get('gender'),
            age               = request.form.get('age'),
            hypertension      = request.form.get('hypertension'),
            heart_disease     = request.form.get('heart_disease'),
            ever_married      = request.form.get('ever_married'),
            work_type         = request.form.get('work_type'),
            residence_type    = request.form.get('Residence_type'),
            avg_glucose_level = request.form.get('avg_glucose_level'),
            bmi               = request.form.get('bmi'),
            smoking_status    = request.form.get('smoking_status'),
        )
    except Exception as e:
        return render_template('home.html', error=f'Input error: {e}')

    proba      = model.predict_proba(x)[0][1]
    percentage = f'{proba * 100:.1f}%'

    if proba >= threshold:
        return render_template(
            'stroke.html',
            pred=f'You have a chance of having a stroke. Probability: {percentage}'
        )
    else:
        return render_template(
            'nostroke.html',
            pred=f'You are safe. Probability of stroke: {percentage}'
        )


if __name__ == '__main__':
    app.run(port=7384, debug=False)
# ── Added by UI redesign ───────────────────────────────────────────────────────
# The old / route served the prediction form directly.
# Now / serves the landing page and /predict serves the multi-step form.
