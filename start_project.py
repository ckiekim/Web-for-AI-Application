from Project import app, load_movie_lr, load_movie_nb, tw_tokenizer
from Project import load_iris, load_mnist

load_movie_lr()
load_movie_nb()
load_iris()
load_mnist()
app.run(host='0.0.0.0', threaded=False)       # 외부 접속 허용시 app.run(host='0.0.0.0') 