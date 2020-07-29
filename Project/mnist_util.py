import os
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist

def mnist_util(app, model, indices):
    (_, _), (X_test, Y_test) = mnist.load_data()
    results = []
    values = []
    for i in range(len(indices)):
        test_data = X_test[indices[i]]
        test_data = test_data.reshape(-1, 28, 28, 1).astype('float32') / 255
        results.append(model.predict_classes(test_data)[0])
        fig, ax = plt.subplots(figsize=(2,2))
        ax.imshow(X_test[indices[i]], cmap='Greys')
        fig.savefig(os.path.join(app.root_path, 'static/images/mnist'+str(i+1)+'.png'))
        values.append(Y_test[indices[i]])

    return results, values
