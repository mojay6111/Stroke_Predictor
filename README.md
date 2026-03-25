# Stroke Predictor

A machine learning web application that predicts a patient's probability of having a stroke based on clinical and lifestyle features.

Built with scikit-learn for the model and Flask for the web interface.

---

## Features

- Predicts stroke probability from 10 clinical inputs
- Web form interface with results page
- User registration and login system
- Email notifications via Gmail SMTP

---

## Project structure

```
Stroke_Predictor/
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   └── 02_model_training.ipynb     # Training, evaluation, model export
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── models/
│   └── model.pkl                   # Trained model (single versioned file)
├── utils/
│   └── preprocessing.py            # Feature encoding shared by training + app
├── templates/                      # HTML pages
├── static/                         # CSS and images
├── app.py                          # Flask application
├── .env.example                    # Environment variable template
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/stroke-predictor.git
cd stroke-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your MySQL and Gmail credentials
```

### 3. Set up the database

Create a MySQL database called `user-system` and run:

```sql
CREATE TABLE user (
    userid INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    password VARCHAR(255)
);
```

### 4. Run the app

```bash
python app.py
```

Visit `http://localhost:7384`

---

## Input features

| Feature | Description |
|---|---|
| Gender | Male / Female / Other |
| Age | Patient age in years |
| Hypertension | 0 = No, 1 = Yes |
| Heart disease | 0 = No, 1 = Yes |
| Ever married | Yes / No |
| Work type | Private / Self-employed / Govt / Children / Never worked |
| Residence type | Urban / Rural |
| Avg glucose level | Average blood glucose (mg/dL) |
| BMI | Body mass index |
| Smoking status | Formerly smoked / Never smoked / Smokes / Unknown |

---

## Dataset

[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle (fedesoriano).

---

## Deployment (Heroku)

```bash
# Make sure Procfile exists:
# web: gunicorn app:app

git push heroku main
```

Set environment variables on Heroku:
```bash
heroku config:set FLASK_SECRET_KEY=your-secret
heroku config:set MAIL_USERNAME=your-email@gmail.com
heroku config:set MAIL_PASSWORD=your-app-password
heroku config:set MYSQL_HOST=your-db-host
heroku config:set MYSQL_USER=your-db-user
heroku config:set MYSQL_PASSWORD=your-db-password
heroku config:set MYSQL_DB=your-db-name
```

---

## License

MIT
