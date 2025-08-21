const express = require('express');
const router = express.Router();
const gemini = require('./../../services/gemini.js');
const Mood = require('./../../infrastructure/db/mood_schema');
router.post('/chat', async (req, res) => {
    try {
        const userMessage = req.body.message;
        const userId = req.body.userid;
        if (!userMessage) {
            return res.status(400).json({ error: 'Message is required' });
        }
        if (!userId) {
            return res.status(400).json({ error: 'User ID is required' });
        }
        const chatbotResponse = await gemini.run(userMessage, userId);
        res.json({ reply: chatbotResponse });
    } catch (error) {
        // console.error('Error during chat:', error);
        res.status(500).json({ error: 'Something went wrong while processing the chat' });
    }
});

router.post('/mood', async (req, res) => {
    try {
        const userid = req.body.userid;
        const latestMood = await Mood.findOne({ userid }).sort({ dateadded: -1 });
        if (latestMood == null) {
            newMood = new Mood({
                id:"0",
                userid:userid,
                feeling_better:false,
                mood:"happy",
                dateadded: Date.now()
            })
            const chatbotResponse = await gemini.getQuote(newMood.mood);
            res.json({ reply: chatbotResponse });
        }
        else{
            const chatbotResponse = await gemini.getQuote(latestMood.mood);
            res.json({ reply: chatbotResponse });
        }
    } catch (error) {
        // console.error('Error during mood:', error);
        res.status(500).json({ reply: 'Something went wrong while processing the mood' });
    }
});

module.exports = router;
