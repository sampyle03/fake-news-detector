Front-End: React.js, html, css
Back-End: JavaScript, Python (v3.11)

# To setup the machine learning model
In order to setup the machine learning model for use, run the data_processor.py and model.py file found at /backend/model_training.

# To run the fake news detector through the webpage
1. Open an integrated terminal for the frontend folder.
2. Run "npm run build" and then close the integrated terminal.
3. Open an integrated terminal for the backend folder.
4. On the integrated terminal, run "npm install"
5. Create a virtual environment using py -3.11 -m venv venv
6. Activate the virtual environment using venv\Scripts\activate.bat
7. In the virtual Environment, run pip install flask flask-cors pandas tensorflow transformers numpy scikit-learn jsonify playwright tf_keras nltk matplotlib shap torch
8. Run "npm run dev"
9. Wait a minute for the Python code to initialize