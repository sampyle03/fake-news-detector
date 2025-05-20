import React from 'react';

const ResultsList = () => {
    
    return (
        <div id="classification-individual-results">
            <ul id="classification-individual-results-list">
                <li class ="classification-individual-result" id="bert-individual-result">
                    <img src="images/BERT.svg" id="bert-individual-result-icon" alt="BERT Classifier Result" />
                    <p id="bert-individual-result-text"></p>
                </li>
                <li class ="classification-individual-result" id="knnc-individual-result">
                    <img src="images/knnc.svg" id="knnc-individual-result-icon" alt="KNN Classifier Result" />
                    <p id="knnc-individual-result-text"></p>
                </li>
                <li class ="classification-individual-result" id="knnr-individual-result">
                    <img src="images/knnr.svg" id="knnr-individual-result-icon" alt="KNNR Classifier Result" />
                    <p id="knnr-individual-result-text"></p>
                </li>
            </ul>
        </div>
    );
}

export default ResultsList;