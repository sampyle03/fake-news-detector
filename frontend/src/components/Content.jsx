import React from 'react';
import Instructions from './Instructions';
import SearchBar from './SearchBar';
import ClassificationResult from './ClassificationResult';

const Content = () => {
    return (
        <div id="main-content">
            <Instructions />
            <SearchBar />
            <ClassificationResult />
        </div>
    );
}

export default Content;