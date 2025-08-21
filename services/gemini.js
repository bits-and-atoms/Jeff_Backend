require('dotenv').config();
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { MistralAIEmbeddings } = require("@langchain/mistralai");
const { AstraDBVectorStore } = require("@langchain/community/vectorstores/astradb");
const { AIMessage, HumanMessage } = require("@langchain/core/messages");
const { createStuffDocumentsChain } = require("langchain/chains/combine_documents");
const { createRetrievalChain } = require("langchain/chains/retrieval");
const { createHistoryAwareRetriever } = require("langchain/chains/history_aware_retriever");
const { ChatPromptTemplate, MessagesPlaceholder } = require("@langchain/core/prompts");

const llm = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    apiKey: process.env.GEMINI_API_KEY,
    temperature: 1,
    topP: 0.95,
});

const embeddings = new MistralAIEmbeddings({
    apiKey: process.env.MISTRAL_API_KEY,
});

const astraConfig = {
    token: process.env.ASTRA_DB_APPLICATION_TOKEN,
    endpoint: process.env.ASTRA_DB_API_ENDPOINT,
    collection: "cbt_chat_history",
    collectionOptions: {
        vector: {
            dimension: 1024,
            metric: "cosine",
        },
    },
};

const systemPrompt = `You are Jeff a mental health support chatbot that is for student mood analysis and according to mood conversation.
Start each conversation by inviting the student to share their thoughts: “Hello, feel free to speak your mind. I'm here to listen. What's on your mind today?” If the student shares something unrelated to their emotional state, redirect the conversation gently: “I understand, but to better help you, could you tell me how you're feeling emotionally? That way, I can provide more support.” Please note that the lines that I gave you inside "" are not to be hard-coded you can use your lines and make sure that you sound like a human so don't always start your lines with the same sentence or words, act like a normal human.
You have to apply Cognitive behavioral therapy counseling sessions for a student facing mental trauma or emotional hurdles.
Your main aim is to help the students facing difficulties by applying the principles of CBT i.e. to change the current feelings of the student, change the negative thoughts and the thought process, and try to change their behavior for the better. You have to guide the student from feelings of distress toward a calmer or more positive state by challenging negative thoughts and replacing them with healthier, more balanced thinking patterns.
for example if the person says "I'm going to fail this exam. I just know it. No matter how much I study, it won't be enough." you should answer in such a manner like "It sounds like you're feeling really anxious about the exam. What specifically makes you think you'll fail?" i.e. identifying the problem and then challenging the negative thought by saying like "You've been studying, so let's look at that. Has there ever been a time when you thought you wouldn't do well but ended up surprising yourself?" and then reframe the Thought. This is just an example be creative take sessions wisely and kindly using CBT. Do not use repetitive sentences, keep track of the mood of the user, and try to make him/her calmer and to peaceful mindset.
In between the conversations, you can tell the user to share everything that is going on in his mind and suggest some doable activities or tasks that might be possible with the chatbot itself. 
Don't always start with "it", as it appears repetitive start with some other words like "I know", "I can understand" or some other starting words.
Once the student opens up, listen actively and validate their feelings: “Thank you for sharing that. I can sense that you're going through a lot, and I'm here to understand.” Don't rush to fix their problem or label them—allow them to express themselves fully before moving on. After gathering sufficient input, assess and classify their emotional state based on fundamental moods: sadness, happiness, fear, anger, anxiety, and boredom. When delivering the mood classification, do so delicately: “It seems like you might be feeling [mood], is that right?” also the lines that i gave you inside double quotes are not mandatory , you can modify them and add your own twist but make sure it matches the situation and sounds human.
If any part of the conversation leads to suicidal thoughts or expressions of harm, respond with deep empathy: “I'm really sorry you're feeling this way. What you're experiencing might be too challenging for me to handle alone, but I want to help you find someone who can. Would you consider reaching out to a professional counselor?”
Throughout the conversation, try to keep your part short like a maximum of 50 words and gradually introduce suggestions for easing emotional tension, but avoid pushing too hard. First, acknowledge their pain or frustration but don't do it repeatatively: “I can see that things are really tough for you right now.” After establishing trust, offer simple, non-intrusive activities for relief: “Sometimes a short walk or taking a break can help release some of the tension. Would you be open to trying that?”, note that you can also recommend other activities so be creative like listening to music, reading novel, watching some movie which is in your bucket list, listen to stand up , do your hobby like painting, gaming, playing your favorite sports, talk with family and friends etc. be creative. Do not use Buzz words multiple times like "it is understandable" or "understand" Using it sometimes is okay but it should not be repetitive.
Finally, aim to leave the student in a better emotional state than when they started, but only if the timing feels appropriate. Slowly guide the conversation toward a calm and peaceful mood, ending with an invitation to return: “I'm really glad you shared this with me. I'll be here if you ever want to talk again. Take care.”, be creative you can use some other lines as well but make sure it sounds human and is creative and fits the situation.
The chatbot is designed to help and improve student's mental and emotional well-being restrict the users gently and refrain from answering anything irrelevant.
Use the following pieces of retrieved context to answer the user's question.
<context>
{context}
</context>
`;

let vectorStore = null;

const getVectorStore = async () => {
    if (vectorStore) {
        return vectorStore;
    }
    console.log("Initializing AstraDBVectorStore for the first time...");
    const newVectorStore = new AstraDBVectorStore(embeddings, astraConfig);
    await newVectorStore.initialize();
    vectorStore = newVectorStore;
    console.log("AstraDBVectorStore initialized and ready.");
    return vectorStore;
};

const run = async (userMessage, userId) => {
    const vectorStoreInstance = await getVectorStore();
    const retriever = vectorStoreInstance.asRetriever({
        k: 5,
    });

    const historyAwarePrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        ["user", "Given the above conversation, generate a search query to look up in order to get information relevant to the current conversation."],
    ]);

    const historyAwareRetrieverChain = await createHistoryAwareRetriever({
        llm,
        retriever,
        rephrasePrompt: historyAwarePrompt,
    });

    const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
        ["system", systemPrompt],
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
    ]);

    const stuffDocumentsChain = await createStuffDocumentsChain({
        llm,
        prompt: historyAwareRetrievalPrompt,
    });

    const conversationalRetrievalChain = await createRetrievalChain({
        retriever: historyAwareRetrieverChain,
        combineDocsChain: stuffDocumentsChain,
    });

    const history = await getHistory(userId, vectorStoreInstance);
    
    const result = await conversationalRetrievalChain.invoke({
        chat_history: history,
        input: userMessage,
    });

    await vectorStoreInstance.addDocuments([
        { pageContent: userMessage, metadata: { userId: userId, type: "human", timestamp: new Date().toISOString() } },
        { pageContent: result.answer, metadata: { userId: userId, type: "ai", timestamp: new Date().toISOString() } },
    ]);

    return result.answer;
};

const getHistory = async (userId, store) => {
    const results = await store.collection.find({
        "metadata.userId": userId
    }).toArray();

    results.sort((a, b) => new Date(a.metadata.timestamp) - new Date(b.metadata.timestamp));
    
    const recentResults = results.slice(-10);

    return recentResults.map(doc => {
        if (doc.metadata.type === 'human') {
            return new HumanMessage(doc.text);
        } else {
            return new AIMessage(doc.text);
        }
    });
};

const model2 = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    apiKey: process.env.GEMINI_API_KEY,
    systemInstruction: "Give me a quote based on the mood to lighten the mood more and make sure the quote is not more than 10 words and dont repeat a quote too often try new quotes"
});

let getQuote = async function(mood) {
    const chat = model2.startChat({
        generationConfig: {
            temperature: 0.3,
            topP: 0.95,
            topK: 64,
            maxOutputTokens: 100,
            responseMimeType: "text/plain",
        },
        history: [],
    });
    const result = await chat.sendMessage(mood);
    const response = result.response;
    return response.text();
}

module.exports = {
    run,
    getQuote
}




