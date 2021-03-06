from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, re, joblib
import numpy as np
import pandas as pd
from PIL import Image
from konlpy.tag import Okt
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg16 import VGG16, decode_predictions
from Project.clu_util import cluster_util   # app.py와 __init__.py에서 다름
from Project.mnist_util import mnist_util
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

app = Flask(__name__)
app.debug = False

vgg = VGG16()
okt = Okt()
movie_lr = None
movie_lr_dtm = None
def load_movie_lr():
    global movie_lr, movie_lr_dtm
    movie_lr = joblib.load(os.path.join(app.root_path, 'model/movie_lr.pkl'))
    movie_lr_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_lr_dtm.pkl'))

def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

movie_nb = None
movie_nb_dtm = None
def load_movie_nb():
    global movie_nb, movie_nb_dtm
    movie_nb = joblib.load(os.path.join(app.root_path, 'model/movie_nb.pkl'))
    movie_nb_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_nb_dtm.pkl'))

def nb_transform(review):
    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    review = review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    morphs = okt.morphs(review, stem=True)
    temp = ' '.join(morph for morph in morphs if not morph in stopwords)
    return temp

model_iris_lr = None
model_iris_svm = None
model_iris_dt = None
model_iris_deep = None
def load_iris():
    global model_iris_lr, model_iris_svm, model_iris_dt, model_iris_deep
    model_iris_lr = joblib.load(os.path.join(app.root_path, 'model/iris_lr.pkl'))
    model_iris_svm = joblib.load(os.path.join(app.root_path, 'model/iris_svm.pkl'))
    model_iris_dt = joblib.load(os.path.join(app.root_path, 'model/iris_dt.pkl'))
    model_iris_deep = load_model(os.path.join(app.root_path, 'model/iris_deep.hdf5'))

model_mnist = None
def load_mnist():
    global model_mnist
    model_mnist = load_model(os.path.join(app.root_path, 'model/mnist-cnn.hdf5'))

@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('regression.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        sp = int(request.form['species'])       # Species
        species = sp_names[sp]
        swid = 0.63711424 * slen - 0.53485016 * plen + 0.55807355 * pwid - 0.12647156 * sp + 0.78264901
        swid = round(swid, 4)
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result.html', menu=menu, iris=iris)

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    menu = {'home':False, 'rgrs':False, 'stmt':True, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('sentiment.html', menu=menu)
    else:
        res_str = ['부정', '긍정']
        review = request.form['review']
        # Logistic Regression 처리
        review_lr = re.sub(r"\d+", " ", review)
        review_lr_dtm = movie_lr_dtm.transform([review_lr])
        result_lr = res_str[movie_lr.predict(review_lr_dtm)[0]]
        # Naive Bayes 처리
        review_nb = nb_transform(review)
        review_nb_dtm = movie_nb_dtm.transform([review_nb])
        result_nb = res_str[movie_nb.predict(review_nb_dtm)[0]]
        # 결과 처리
        movie = {'review':review, 'result_lr':result_lr, 'result_nb':result_nb}
        return render_template('senti_result.html', menu=menu, movie=movie)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification.html', menu=menu)
    else:
        f = request.files['image']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)
        img = np.array(Image.open(filename).resize((224, 224)))
        yhat = vgg.predict(img.reshape(-1, 224, 224, 3))
        label_key = np.argmax(yhat)
        label = decode_predictions(yhat)
        label = label[0][0]
        return render_template('cla_result.html', menu=menu,
                                filename = secure_filename(f.filename),
                                name=label[1], pct='%.2f' % (label[2]*100))

@app.route('/classification_iris', methods=['GET', 'POST'])
def classification_iris():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_iris.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Sepal Width
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        test_data = np.array([slen, swid, plen, pwid]).reshape(1,4)
        species_lr = sp_names[model_iris_lr.predict(test_data)[0]]
        species_svm = sp_names[model_iris_svm.predict(test_data)[0]]
        species_dt = sp_names[model_iris_dt.predict(test_data)[0]]
        species_deep = sp_names[model_iris_deep.predict_classes(test_data)[0]]
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 
                'species_lr':species_lr, 'species_svm':species_svm,
                'species_dt':species_dt, 'species_deep':species_deep}
        return render_template('cla_iris_result.html', menu=menu, iris=iris)

@app.route('/classification_mnist', methods=['GET', 'POST'])
def classification_mnist():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_mnist.html', menu=menu)
    else:
        idx1 = int(request.form['idx1'])
        idx2 = int(request.form['idx2'])
        idx3 = int(request.form['idx3'])
        indices = [idx1, idx2, idx3]
        results, values = mnist_util(app, model_mnist, indices)
        mnist_file = os.path.join(app.root_path, 'static/images/mnist1.png')
        mtime = int(os.stat(mnist_file).st_mtime)
        return render_template('cla_mnist_result.html', menu=menu, mtime=mtime,
                                indices=indices, results=results, values=values)

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':True, 'user':False}
    if request.method == 'GET':
        return render_template('clustering.html', menu=menu)
    else:
        f = request.files['csv']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)
        ncls = int(request.form['K'])
        cluster_util(app, ncls, secure_filename(f.filename))
        img_file = os.path.join(app.root_path, 'static/images/kmc.png')
        mtime = int(os.stat(img_file).st_mtime)
        return render_template('clu_result.html', menu=menu, K=ncls, mtime=mtime)

@app.route('/member/<name>')
def member(name):
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':True}
    nickname = request.args.get('nickname', '별명: 없음')
    return render_template('user.html', menu=menu, name=name, nickname=nickname)

@app.route('/shutdown')
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

# app.py와 __init__.py에서 다름
''' if __name__ == '__main__':
    load_movie_lr()
    load_movie_nb()
    load_iris()
    load_mnist()
    app.run() '''     # 외부 접속 허용시 host='0.0.0.0' 추가