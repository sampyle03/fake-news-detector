# X Fake News Detection System

Front-End: React.js, html, css
Back-End: JavaScript, Python (v3.11)

## To setup the machine learning model
In order to setup the machine learning model for use:
1. Ensure you are using Python 3.11 as your Python interpreter. Install it here: https://www.python.org/downloads/release/python-3117/
2. On your Windows machine, click the bottom left Windows icon and type in "cmd" to open a terminal.
3. (Optional to save memory) - type in the following command:
    > cd C:\Users\user\path\to\folder\fake-news-detector-main
4. On the terminal, run the line of code:
    > py -3.11 -m pip install flask flask-cors pandas tensorflow transformers numpy scikit-learn jsonify playwright tf_keras nltk matplotlib shap torch
<br />

5. Navigate to the data_processor.py file found at /backend/model_training.
6. Uncomment all lines of code below TRAINING DATA - you can uncomment lines by highlighting the required code and clicking "Ctrl + /".
7. Run the data_processor.py file and follow the prompts (IMPORTANT: all process_statements() functions should remain commented) (WARNING: this will take a few hours to run).
8. Re-comment all lines of code that you had just commented out - you can recomment lines by highlighting the required code and clicking "Ctrl + /".
<br />

9. Navigate to the bert_classifier.py file found at /backend/model_training.
10. Uncomment the top line of the bottom three lines of code and follow the prompts. You can uncomment lines by highlighting the required code and pressing "Ctrl + /".
11. (Optional) For testing, uncomment all 3 and follow the prompts.
12. Run the bert_classifier.py file (WARNING: this may take a few hours to run).
13. Re-comment all lines of code that you had just commented out - you can recomment lines by highlighting the required code and clicking "Ctrl + /".
<br />

14. Navigate to the model.py file found at /backend/model_training.
15. Uncomment all lines of code below "# Main - uncomment to run" - you can uncomment lines by highlighting the required code and clicking "Ctrl + /".
16. Run the model.py file. This could take up to 10 minutes to run.
17. Re-comment all lines of code that you had just commented out - you can recomment lines by highlighting the required code and clicking "Ctrl + /".
<br />

18. Navigate to the tweet_scraper.py file and run it once. This may take up to a minute to run.
19. Navigate to the ensemble.py file and run it once. This may take up to a minute to run.

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
