from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
#initial training with random dataset#
print("Running new main.py")
np.random.seed(42)
n_samples = 200
income = np.random.randint(2000, 10000, n_samples)
loan = np.random.randint(100, 700, n_samples)
credit = np.random.randint(0, 2, n_samples)

# Logic for approval probability
y = ((income / loan) > 10).astype(int)
y = (y * 0.7 + credit * 0.3 + np.random.randn(n_samples) * 0.1 > 0.5).astype(int).reshape(-1, 1)

x = np.vstack([income, loan, credit]).T
x_max = np.max(x, axis=0)
x = x / x_max

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

x=np.array(x)
y=np.array(y).reshape(-1,1)
np.random.seed(42)
n=x.shape[1]
w=np.random.randn(n,1)
b=np.random.randn(1)

lr=0.1
#training part#
for epoch in range(10000):
    z=np.dot(x,w)+b
    a=sigmoid(z)
    loss = np.mean((y - a)**2)
    dz=2*(a-y)*deriv_sigmoid(z)
    dw=np.dot(x.T,dz)/len(x)
    db=np.mean(dz)
    
    w-=lr*dw
    b-=lr*db

def predict(features):
    features = features / x_max
    prob = (np.dot(features, w) + b)
    return sigmoid(prob)[0][0]
#function for graph plotting#
def feature_contribution_plot(features, W):
    
    contributions = (features / np.max(x, axis=0)) * W.flatten()  
    labels = ['Income','Loan Amount','Credit History']
    contributions = contributions.flatten() 
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(labels, contributions, color=['blue','orange','green'])
    ax.set_ylabel('Contribution to Prediction')
    ax.set_title('Feature Contribution')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_base64

app=Flask(__name__)
app.secret_key="secret123"
#initialize the database#
def init_db():
    conn=sqlite3.connect("data.db")
    c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
        )''')
    c.execute('''CREATE TABLE IF NOT EXISTS loan(
        name TEXT,income REAL, loan REAL, credit_score INTEGER, result TEXT
        )''')
    for i, (income, loan, credit) in enumerate(x):
        result = "approved" if y[i] == 1 else "rejected"
        c.execute(
            "INSERT INTO loan (name, income, loan, credit_score, result) VALUES (?, ?, ?, ?, ?)",
            (f"User_{i+1}", income, loan, credit, result)
            )
    conn.commit()
    conn.close()
    
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('login'))
    return redirect(url_for('login'))

@app.route('/register',methods=['POST','GET'])
def register():
    message='Kindly register!'
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        
        conn=sqlite3.connect("data.db")
        c=conn.cursor()
        
        c.execute("SELECT * FROM users WHERE username=?",(username,))
        existing_user=c.fetchone()
        if existing_user:
            message="user already exists"
        else:
            message="welcome to the login page"
            c.execute("INSERT INTO users (username,password) VALUES (?,?)",(username,password))
            conn.commit()
            conn.close()
            return redirect(url_for('apply'))
    return render_template('register.html',message=message)
    
@app.route('/login',methods=['POST','GET'])
def login():
    message='login if you have already registerd'
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        conn=sqlite3.connect("data.db")
        c=conn.cursor()
        c.execute("SELECT * FROM users where username=? and password=?",(username,password))
        user=c.fetchone()
        c.close()

        if user:
            session['username']=username
            return redirect(url_for('apply'))
        else:
            message="invalid login credentials..."
    return render_template('login.html',message=message)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/apply',methods=['POST','GET'])
def apply():
    if request.method=="POST":
        name=request.form['name']
        income=float(request.form['income'])
        credit=int(request.form['credit'])
        loan=float(request.form['loan'])
        features = np.array([[income, loan, credit]])
        pred = float(predict(features.reshape(1, -1)))
        result='approved' if pred>0.5 else 'rejected'
        interest_rate = round(20 - (pred * 10), 2)
        if interest_rate < 5:
            interest_rate = 5 
        
        conn=sqlite3.connect("data.db")
        c=conn.cursor()
        c.execute("insert into loan (name,income,loan,credit_score,result) values(?,?,?,?,?)",(name,income,loan,credit,result))
        try:
            plot = feature_contribution_plot(features, w)
        except Exception as e:
            print("Plot error:", e)
            plot = None
        conn.commit()
        conn.close()
        return render_template('result.html',plot=plot,result=result,pred=sigmoid(pred),interest_rate=interest_rate)
    return render_template('apply.html')
        
@app.route('/admin')
def admin():
    conn=sqlite3.connect("data.db")
    c=conn.cursor()
    c.execute('select * from loan')
    rows=c.fetchall()
    conn.close()
    return render_template('admin.html',rows=rows)

if __name__=="__main__":
    init_db()
    app.run(port=5500,debug=True, use_reloader=False)
    
    