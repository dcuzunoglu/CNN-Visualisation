"""
Chart generation script

"""
import os

from flask import Flask, request
from flask import render_template

from convnet import calculate_activations_for_image_and_predict
from constants import FLOWERS_RESIZED_DIR
app = Flask(__name__, template_folder='templates')


@app.route("/cnn_visualise_activations/")
def cnn_webapp_main_page():
    """
    Bar chart in Flask
    :return:
    """
    all_images = sorted(os.listdir(FLOWERS_RESIZED_DIR))

    return render_template('cnn_webapp_main_page.html', all_images=all_images)


@app.route("/convnet/", methods=['POST'])
def run_convnet():
    """
    Function to interface with convnet
    :return:
    """

    # Get the requested image name from AJAX request
    data = request.form
    image_name = data["imageName"]

    # Run convnet model on following image
    output_path = calculate_activations_for_image_and_predict(image_name, from_flask=True)
    return output_path


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)



