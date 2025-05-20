import React from 'react';

const SearchBar = () => {

    const makeSearch = (event) => {
        event.preventDefault();

        // render loading gif
        const loadingGif = document.getElementById("loading-gif"); 
        loadingGif.style.display = "block";

        // hide classification results of previous search
        document.getElementById("classification-total-result").style.display = "none";
        document.getElementById("classification-confidence").style.display = "none";
        document.getElementById("classification-individual-results").style.display = "none";
        document.getElementById("top-3-tokens").style.display = "none";


        const url = document.getElementById("home-search-bar").value;
        if (url === "") {
            alert("Please enter a URL.");
            return;
        }
        const regex = /^https:\/\/x\.com\/[^\/]+\/status\/\d+$/; // regex to ensure the url is a valid x.com post (status)
;
        if (!regex.test(url)) {
            alert("Please enter a valid URL.");
            return;
        }
        // console.log("Making request to classify URL:", url);
        
        // get classification from backend
        const response = fetch("/classify", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: url })
        }).then((response) => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        }).then((data) => {
            //console.log("Response data:", data);
            const prediction = data.prediction;
            const individual_predictions = data.individual_predictions;
            const confidence = data.confidence;
            const top3Tokens = data.top_3_tokens;
            //console.log("Top 3 Tokens:", top3Tokens);
            // Ensure response contains expected data
            if (prediction === undefined || individual_predictions === undefined || confidence === undefined || top3Tokens === undefined) {
                console.error("There was an error with the response data.");
                return;
            }
            // Hide loading gif
            const loadingGif = document.getElementById("loading-gif");
            loadingGif.style.display = "none";

            try {
                // Display total classification result and confidence
                const classificationTotalResult = document.getElementById("classification-total-result");
                const classificationConfidence = document.getElementById("classification-confidence");
                if (prediction === 0) {
                    classificationTotalResult.style.color = "red";
                    classificationTotalResult.innerHTML = "PREDICTION: FAKE NEWS";
                } else if (prediction === 1) {
                    classificationTotalResult.style.color = "green";
                    classificationTotalResult.innerHTML = "PREDICTION: REAL NEWS";
                } else {
                    classificationTotalResult.style.color = "grey";
                    classificationTotalResult.innerHTML = "This tweet cannot be classified.";
                }
                classificationConfidence.innerHTML = "Confidence: " + confidence;
                classificationConfidence.style.display = "block";
                classificationTotalResult.style.display = "block";

                // Display individual classification results
                const bertResultIcon = document.getElementById("bert-individual-result-icon");
                const knncResultIcon = document.getElementById("knnc-individual-result-icon");
                const knnrResultIcon = document.getElementById("knnr-individual-result-icon");
                const bertResultText = document.getElementById("bert-individual-result-text");
                const knncResultText = document.getElementById("knnc-individual-result-text");
                const knnrResultText = document.getElementById("knnr-individual-result-text");
                if (individual_predictions[0] < 0.43) {
                    bertResultIcon.classList.add("red");
                    bertResultText.innerHTML = "Model 1 (BERT) predicts that this post is fake news.<br />BERT is the most accurate fake news predictor.";
                } else {
                    bertResultIcon.classList.add("green");
                    bertResultText.innerHTML = "Model 1 (BERT) predicts that this post is real news.\nBERT is the most accurate fake news predictor.";
                }
                if (individual_predictions[1] < 0.32) {
                    knncResultIcon.classList.add("red");
                    knncResultText.innerHTML = "Model 2 (KNN Classifier) predicts that this post is fake news.";
                } else {
                    knncResultIcon.classList.add("green");
                    knncResultText.innerHTML = "Model 2 (KNN Classifier) predicts that this post is real news.";
                }
                if (individual_predictions[2] < 0.41) {
                    knnrResultIcon.classList.add("red");
                    knnrResultText.innerHTML = "Model 3 (KNN Regressor) predicts that this post is fake news.";
                } else {
                    knnrResultIcon.classList.add("green");
                    knnrResultText.innerHTML = "Model 3 (KNN Regressor) predicts that this post is fake news.";
                }
                if (individual_predictions[0] !== undefined && individual_predictions[1] !== undefined && individual_predictions[2] !== undefined) {
                    let individualPredictionsHTML = document.getElementById("classification-individual-results");
                    individualPredictionsHTML.style.display = "block";
                }

                // Display top 3 tokens that contributed to the classification
                const token1 = document.getElementById("top-3-token-1-text");
                const token2 = document.getElementById("top-3-token-2-text");
                const token3 = document.getElementById("top-3-token-3-text");
                if (top3Tokens[0][0] > 0) {
                    token1.style.color = "green";
                } else {
                    token1.style.color = "red";
                }
                if (top3Tokens[1][0] > 0) {
                    token2.style.color = "green";
                } else {
                    token2.style.color = "red";
                }
                if (top3Tokens[2][0] > 0) {
                    token3.style.color = "green";
                } else {
                    token3.style.color = "red";
                }
                token1.innerHTML = "Token 1: " + top3Tokens[0][1];
                token2.innerHTML = "Token 2: " + top3Tokens[1][1];
                token3.innerHTML = "Token 3: " + top3Tokens[2][1];
                const top3TokensHTML = document.getElementById("top-3-tokens");
                top3TokensHTML.style.display = "block"; // display top 3 tokens
            } catch (error) {
                // Handle errors that occur during UI update
                console.error("Error updating the UI:", error);
                alert("There was an error updating the UI. Please try again later.");
            }
        }).catch((error) => {
            // error handling for fetch request
            console.error("There was a problem with the fetch operation:", error);
            alert("There was an error processing your request. Please try again later.");
        });
    }
    
    return (
        <>
            <form id="home-search-form">
                <input id="home-search-bar" type="text" placeholder="https://x.com/UEA_CMP/status/1835978986913988995" />
                <button id="home-detect-button" onClick={makeSearch}>Detect</button>
            </form>
        </>
    );
}

export default SearchBar;