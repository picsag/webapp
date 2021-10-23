from flask import Flask, render_template
from flask import jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import numpy as np
from numpy.random import randn

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# plot the generated images
def create_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n,n,1+i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :])
    pyplot.savefig('./static/images/plot.png')

def inference():
    # load model
    model = load_model('generator_model_250.h5')

    latent_points = generate_latent_points(100, 100)
    # generate images
    X = model.predict(latent_points)
    X = (X + 1) / 2.0
    create_plot(X, 10)

application = Flask(__name__)

@application.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    print("I am inside hello world")
    return 'Hello World! CD'

@application.route('/echo/<name>')
def echo(name):
    print(f"This was placed in the url: new-{name}")
    val = {"new-name": name}
    return jsonify(val)

@application.route('/gan_cifar10')
def gan_cifar10():
    print("Running GAN . . .")
    inference()
    return render_template('plot.html', url='./static/images/plot.png')


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080, debug=True)