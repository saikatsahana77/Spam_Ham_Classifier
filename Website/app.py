from flask import Flask
from flask import render_template,request,redirect,url_for
from predict import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    val = request.form['value']
    prediction = predict(val)
    prediction = prediction[0]
    return render_template('results.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)