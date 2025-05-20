import React from 'react';

const ShapResults = () => {
    
    return (
        <div id="top-3-tokens">
            <h2>Top 3 Tokens</h2> 
            <p> Tokens are words, terms or punctuation, or even sub-words that the post was broken into.<br /> These tokens are the most important for the classification of the URL. Green tokens suggest that the post is real, while red tokens suggest that the post is fake.</p>
            <ul id="top-3-tokens-list">
                <li className="top-3-token" id="top-3-token-1">
                    <h3 class="top-3-token-h3" id="top-3-token-1-text"></h3>
                </li>
                <li className="top-3-token" id="top-3-token-2">
                    <h3 class="top-3-token-h3" id="top-3-token-2-text"></h3>
                </li>
                <li className="top-3-token" id="top-3-token-3">
                    <h3 class="top-3-token-h3" id="top-3-token-3-text"></h3>
                </li>
            </ul>
        </div>
    );
}

export default ShapResults;