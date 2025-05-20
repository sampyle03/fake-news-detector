import React from 'react';
import ResultsList from './ResultsList';
import ShapResults from './ShapResults';

const ClassificationResult = () => {
    
    return (
        <>
            <img src="images/loading.gif" id="loading-gif" alt="Loading..." />
            <h3 id="classification-total-result"></h3>
            <h4 id="classification-confidence"></h4>
            <ResultsList />
            <ShapResults />
        </>
    );
}

export default ClassificationResult;