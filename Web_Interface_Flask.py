from flask import Flask, render_template, send_file, redirect, url_for, request
from werkzeug.utils import secure_filename
import os
from functions_used_in_web_interface import run_model

# Define the root directory of your project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Now you can use ROOT_DIR to build the full path to your files
file_path = os.path.join(ROOT_DIR, 'Github_folders', 'Web_Interface_Flask.py')


app = Flask(__name__)

# Set of allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file_and_run_model():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            directory = os.path.join(ROOT_DIR, 'Images_to_be_used')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filepath = os.path.join(directory, filename)
            file.save(filepath)

            # Now you can use the uploaded image in your FINAL_MODEL code
            output_image_path = run_model(filepath)
            # Check if run_model completed successfully
            if output_image_path is None:
                return "Error: An error occurred in run_model."
            # Check if the file exists before sending it
            elif os.path.exists(output_image_path):
                return send_file(output_image_path, mimetype='image/png')
            else:
                return "Error: The output image file does not exist."

@app.route('/success')
def success():
    return '''
    <h1>Image uploaded and processed successfully!</h1>
    <a href="/">Return to homepage</a>
    '''

if __name__ == '__main__':
    app.run(debug = True)