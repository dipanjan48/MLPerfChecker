from flask import Flask,render_template,request
import pandas as pd
import csv
from preprocessing import preprocessing
from model import ModelSelection
import os


app = Flask(__name__)

@app.route('/' , methods = ['GET' , 'POST'])
def index():
    return render_template('homepage.html')

@app.route('/data', methods = ['GET' , 'POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        if f is None:
            pass
        else:
            dep_var1 = request.form['dep_var']
            df = pd.read_csv(f)
            preproc = preprocessing(df,dep_var1)
            X_train, X_test , Y_train , Y_test = preproc.split()
            ModelSel = ModelSelection(X_train, X_test , Y_train , Y_test)
            cm, precision_score1 , recall_score1 = ModelSel.ModelPerformance()
            #df_return = encode_independent_var.
            #df_q = df_return.shape
            return (render_template('homepage.html', precision=precision_score1*100 , recall=recall_score1*100))


'''@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])

def download():
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename)'''


if __name__ == '__main__':
    app.run(debug = True)

