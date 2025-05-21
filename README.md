# X Fake News Detection System

Front-End: React.js, html, css
Back-End: JavaScript, Python (v3.11)

## To setup the machine learning model
In order to setup the machine learning model for use:
1. Ensure you are using Python 3.11 as your Python interpreter. Install it here: https://www.python.org/downloads/release/python-3117/
2. On your Windows machine, click the bottom left Windows icon and type in "cmd" to open a terminal.
3. On the terminal, run the line of code:
    > py -3.11 -m pip install flask flask-cors pandas tensorflow transformers numpy scikit-learn jsonify playwright tf_keras nltk matplotlib shap torch
<br />

4. Navigate to the data_processor.py file found at /backend/model_training.
5. Uncomment all lines of code below TRAINING DATA.
6. Run the data_processor.py file (WARNING: this will take a few hours to run).
7. Re-comment all lines of code that you had just commented out.
<br />

8. Navigate to the bert_classifier.py file found at /backend/model_training.
9. Uncomment the top line of the bottom 3 lines of code. For testing, uncomment all 3 and follow prompts.
10. Run the bert_classifier.py file (WARNING: this may take a few hours to run).
11. Re-comment all lines of code that you had just commented out.
<br />

12. Navigate to the model.py file found at /backend/model_training.
13. Uncomment all lines of code below "# Main - uncomment to run"
14. Run the model.py file. This could take up to 10 minutes to run.
15. Re-comment all lines of code that you had just commented out.
<br />

16. Navigate to the tweet_scraper.py file and run it once. This may take up to a minute to run.
17. Navigate to the ensemble.py file and run it once. This may take up to a minute to run.

## To run the fake news detector through the webpage
1. Open an integrated terminal for the frontend folder. This can be done through a command such as:
    > cd C:\Users\user1\Documents\FakeNewsDetector\Code\frontend
2. Run the following command:
    > npm run build
3. Close the integrated terminal.
3. Open an integrated terminal for the backend folder. This can be done through a command such as:
    > cd C:\Users\user1\Documents\FakeNewsDetector\Code\backend
4. On the integrated terminal, run the following command:
    > npm install
5. Create a virtual environment using the following command:
    >  py -3.11 -m venv venv
6. Activate the virtual environment using the following command:
    > venv\Scripts\activate.bat
7. In the virtual Environment, run the following command:
    > pip install flask flask-cors pandas tensorflow transformers numpy scikit-learn jsonify playwright tf_keras nltk matplotlib shap torch
8. Run the following command
    > npm run dev
9. Wait a minute for the Python code to initialize.
10. Once the command line returns with a debugger pin such as Debugger PIN: 123-456-789, open your chosen browser.
11. In your browser, type in
    > localhost:3000

## To use the webpage
1. Paste the URL of an X/Twitter post into the search bar.
2. Click "Submit" and wait for the results to appear - this may take up to a minute.
3. Repeat steps 1 and 2.