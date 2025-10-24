
💰 Loan Approval Prediction System

A Flask-based web application that allows users to register, log in, and apply for a loan, with the approval decision determined using a Machine Learning model trained on synthetic financial data.

🚀 Project Overview

This project demonstrates a complete end-to-end system that integrates:

Machine Learning — Logistic Regression–like model built from scratch using NumPy

Flask — Backend web framework for routing and API handling

SQLite — Lightweight database for user and loan record storage

HTML & CSS — Frontend design for user interface

Matplotlib — Visualization of feature contributions influencing the prediction

Users can:

Register and log in securely.

Fill in loan application details (income, loan amount, credit history).

Get immediate feedback — Approved or Rejected — with an interest rate and feature importance graph.

🧠 Machine Learning Component

The ML part uses a custom-implemented logistic regression model trained using gradient descent.
It predicts the probability of loan approval based on 3 key features:

Feature	Description
Income	Applicant’s monthly income
Loan Amount	Amount requested for loan
Credit History	Binary indicator (1 = good credit, 0 = poor credit)
🔹 Training Details

Model weights (w) and bias (b) are trained on a synthetic dataset generated using NumPy.

The model learns the relationship between income, loan amount, and credit score.

Training uses the Mean Squared Error as loss and Sigmoid activation to output probabilities.

🔹 Prediction Logic

After training:

prob = sigmoid(np.dot(features, w) + b)


If prob > 0.5 → Loan Approved, else Rejected.
The interest rate is inversely proportional to this predicted probability.

🔹 Visualization

A feature contribution plot is generated using Matplotlib, showing how each feature (income, loan, credit) influenced the model’s output.

⚙️ Tech Stack
Layer	Technology
Frontend	HTML5, CSS3
Backend	Flask (Python)
Database	SQLite
Machine Learning	NumPy
Visualization	Matplotlib
🗂️ Project Structure
loan-approval/
│
├── main.py                  # Main Flask application + ML model
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── apply.html
│   ├── result.html
│   └── admin.html
├── data.db                  # SQLite database (auto-created)
├── static/                  # Optional (for images/CSS)
└── README.md                # Project documentation
