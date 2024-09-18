const express = require('express');
const OpenAIApi = require('openai');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());
app.use(express.static('public')); // Serve static files from 'public' directory

const openai = new OpenAIApi({
  apiKey: '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',
});

app.post('/ask', async (req, res) => {
    try {
        const response = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [{ role: 'user', content: req.body.query }],
        });
        console.log("OpenAI API response:", response);
        const answer = response.choices[0].message.content.trim();
        res.json({ answer: answer });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'An error occurred', details: error.message });
    }
});



const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
