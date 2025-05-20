const express = require("express");
var bodyParser = require("body-parser");

const app = express();
const port = 3000;
const path = require('path');
const { url } = require("inspector");

app.use(express.json());
app.use(express.static(path.join(__dirname, "../frontend/dist")));

// use dist folder for static files - React build
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/dist/index.html"));
});

async function classifyURL(url) {
    // POST request to backend flask server (port 5000) to classify the URL using ensemble model
    const res = await fetch("http://localhost:5000/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
    });
    // /classify does return jsonify({"prediction": prediction, "individual_predictions": individual_predictions})
    const { prediction, individual_predictions, confidence, top_3_tokens } = await res.json();
    
    // Check if the response contains the expected data
    if (prediction === undefined || individual_predictions === undefined || confidence === undefined || top_3_tokens === undefined) {
        console.error("Error: prediction, individual_predictions, confidence, or top_3_tokens is undefined");
        return null;
    }
    return { prediction, individual_predictions, confidence, top_3_tokens };
}

app.post('/classify', async (req,res) => {
    let url = req.body.url;
    let prediction, individual_predictions, confidence, top_3_tokens;
    try {
        ({ prediction, individual_predictions, confidence, top_3_tokens } = await classifyURL(url));
    } catch (error) {
        console.error("Error classifying URL:", error);
        return res.status(500).json({ error: "Internal Server Error" });
    }
    if (prediction === null) {
        return res.status(500).json({ error: "Error classifying URL" });
    }
    res.json({ prediction, individual_predictions, confidence, top_3_tokens });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});