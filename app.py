import os
import io
import joblib
from datetime import datetime

import numpy as np
import MySQLdb.cursors
from dotenv import load_dotenv
from flask import (Flask, render_template, request, redirect,
                   url_for, session, send_file)
from flask_mail import Mail, Message
from flask_mysqldb import MySQL
from weasyprint import HTML

load_dotenv()

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-dev-key-change-in-prod')

app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 465
app.config['MAIL_USE_SSL']  = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(app)

app.config['MYSQL_HOST']     = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER']     = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB']       = os.getenv('MYSQL_DB', 'stroke_predictor')
mysql = MySQL(app)

# ── Model ──────────────────────────────────────────────────────────────────────
MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
model       = joblib.load(os.path.join(MODELS_PATH, 'model.pkl'))
threshold   = joblib.load(os.path.join(MODELS_PATH, 'threshold.pkl'))

from utils.preprocessing import preprocess_input

# ── Helpers ────────────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('loggedin'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_risk_tips(risk_level):
    if risk_level == 'high':
        return [
            'Schedule an appointment with your doctor immediately',
            'Have your blood pressure and glucose checked professionally',
            'Discuss your risk factors and medical history with a specialist',
            'Make lifestyle changes: quit smoking, reduce salt, exercise more',
            'Learn to recognise the F.A.S.T warning signs of stroke',
        ]
    return [
        'Maintain regular blood pressure and glucose checks',
        'Stay physically active — at least 30 minutes most days',
        'Eat a balanced diet low in salt and saturated fats',
        'Avoid smoking and limit alcohol consumption',
        'Manage stress through relaxation and good sleep',
    ]

# ── Routes: Public ─────────────────────────────────────────────────────────────
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

# ── Routes: Auth ───────────────────────────────────────────────────────────────
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
            return redirect(url_for('predict'))
        else:
            message = 'Incorrect email or password.'
    return render_template('login.html', message=message)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ── Routes: Profile ────────────────────────────────────────────────────────────
@app.route('/profile')
@login_required
def profile():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM user WHERE userid = %s', (session['userid'],))
    user = cursor.fetchone()

    cursor.execute(
        'SELECT * FROM assessments WHERE user_id = %s ORDER BY assessed_at DESC LIMIT 5',
        (session['userid'],)
    )
    recent = cursor.fetchall()

    cursor.execute(
        'SELECT COUNT(*) as total FROM assessments WHERE user_id = %s',
        (session['userid'],)
    )
    total = cursor.fetchone()['total']

    cursor.execute(
        'SELECT COUNT(*) as high FROM assessments WHERE user_id = %s AND risk_level = %s',
        (session['userid'], 'high')
    )
    high_count = cursor.fetchone()['high']

    return render_template('profile.html',
                           user=user,
                           recent=recent,
                           total=total,
                           high_count=high_count)


@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def profile_edit():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    message = ''
    if request.method == 'POST':
        name   = request.form.get('name', '').strip()
        age    = request.form.get('age', '')
        gender = request.form.get('gender', '')
        phone  = request.form.get('phone', '').strip()
        cursor.execute(
            'UPDATE user SET name=%s, age=%s, gender=%s, phone=%s WHERE userid=%s',
            (name, age or None, gender or None, phone or None, session['userid'])
        )
        mysql.connection.commit()
        session['name'] = name
        message = 'Profile updated successfully.'

    cursor.execute('SELECT * FROM user WHERE userid = %s', (session['userid'],))
    user = cursor.fetchone()
    return render_template('profile_edit.html', user=user, message=message)


@app.route('/profile/change-password', methods=['POST'])
@login_required
def change_password():
    current  = request.form.get('current_password', '')
    new_pw   = request.form.get('new_password', '')
    confirm  = request.form.get('confirm_password', '')
    cursor   = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute(
        'SELECT * FROM user WHERE userid = %s AND password = %s',
        (session['userid'], current)
    )
    if not cursor.fetchone():
        return redirect(url_for('profile_edit') + '?pw_error=Incorrect+current+password')
    if new_pw != confirm:
        return redirect(url_for('profile_edit') + '?pw_error=Passwords+do+not+match')

    cursor.execute(
        'UPDATE user SET password = %s WHERE userid = %s',
        (new_pw, session['userid'])
    )
    mysql.connection.commit()
    return redirect(url_for('profile_edit') + '?pw_success=Password+changed+successfully')


@app.route('/history')
@login_required
def history():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT * FROM assessments WHERE user_id = %s ORDER BY assessed_at DESC',
        (session['userid'],)
    )
    assessments = cursor.fetchall()
    return render_template('history.html', assessments=assessments)


# ── Routes: Prediction ─────────────────────────────────────────────────────────
@app.route('/result', methods=['POST'])
def result():
    try:
        form = request.form
        x = preprocess_input(
            gender            = form.get('gender'),
            age               = form.get('age'),
            hypertension      = form.get('hypertension'),
            heart_disease     = form.get('heart_disease'),
            ever_married      = form.get('ever_married'),
            work_type         = form.get('work_type'),
            residence_type    = form.get('Residence_type'),
            avg_glucose_level = form.get('avg_glucose_level'),
            bmi               = form.get('bmi'),
            smoking_status    = form.get('smoking_status'),
        )
    except Exception as e:
        return render_template('home.html', error=f'Input error: {e}')

    proba      = model.predict_proba(x)[0][1]
    percentage = f'{proba * 100:.1f}%'
    risk_level = 'high' if proba >= threshold else 'low'
    tips       = get_risk_tips(risk_level)

    # Save to history if logged in
    if session.get('loggedin'):
        try:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('''
                INSERT INTO assessments
                (user_id, gender, age, hypertension, heart_disease, ever_married,
                 work_type, residence_type, avg_glucose_level, bmi,
                 smoking_status, stroke_probability, risk_level)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ''', (
                session['userid'],
                form.get('gender'), form.get('age'),
                form.get('hypertension'), form.get('heart_disease'),
                form.get('ever_married'), form.get('work_type'),
                form.get('Residence_type'), form.get('avg_glucose_level'),
                form.get('bmi'), form.get('smoking_status'),
                round(proba * 100, 2), risk_level
            ))
            mysql.connection.commit()
            # Get the assessment id for PDF link
            assessment_id = cursor.lastrowid
        except Exception:
            assessment_id = None
    else:
        assessment_id = None

    template = 'stroke.html' if risk_level == 'high' else 'nostroke.html'
    return render_template(template,
                           pred=f'Probability of stroke: {percentage}',
                           percentage=percentage,
                           proba=round(proba * 100, 1),
                           risk_level=risk_level,
                           tips=tips,
                           assessment_id=assessment_id,
                           form_data=dict(form))


@app.route('/result/pdf/<int:assessment_id>')
@login_required
def result_pdf(assessment_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'SELECT a.*, u.name, u.email, u.age as profile_age, u.gender as profile_gender '
        'FROM assessments a JOIN user u ON a.user_id = u.userid '
        'WHERE a.id = %s AND a.user_id = %s',
        (assessment_id, session['userid'])
    )
    assessment = cursor.fetchone()
    if not assessment:
        return redirect(url_for('history'))

    tips = get_risk_tips(assessment['risk_level'])
    html_string = render_template('pdf_report.html',
                                  assessment=assessment,
                                  tips=tips,
                                  generated_at=datetime.now().strftime('%B %d, %Y at %H:%M'))
    pdf_bytes = HTML(string=html_string, base_url=request.host_url).write_pdf()
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'stroke_report_{assessment_id}.pdf'
    )


if __name__ == '__main__':
    app.run(port=7384, debug=False)
