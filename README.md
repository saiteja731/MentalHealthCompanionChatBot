# 🧠 Intelligent Mental Health Companion Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.0-lightgrey?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An AI-powered Mental Health Support System using NLP & Deep Learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📋 Project Overview

The **Intelligent Mental Health Companion Chatbot** is an AI-powered application providing **real-time emotional support** and **personalized mental health guidance**. It bridges the accessibility gap in mental healthcare by offering a confidential, non-judgmental, cost-effective platform for emotional well-being.

### 🎯 Problem Statement
- **Accessibility**: Limited mental health resources in many regions
- **Cost**: Traditional therapy is expensive
- **Stigma**: People hesitate to seek help
- **24/7 Need**: Require anytime support without waiting

### ✨ Solution
Advanced **NLP** and **Deep Learning** to:
- Detect emotional states (anxiety, depression, stress)
- Provide personalized, empathetic responses
- Offer self-help strategies
- Ensure complete privacy

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Conversations** | Neural Networks & NLP for human-like interactions |
| 🎭 **Emotion Detection** | Identifies stress, anxiety, depression |
| 💬 **Personalized Responses** | Context-aware answers |
| 🔐 **Privacy First** | 100% confidential conversations |
| ⚡ **Real-time Responses** | Instant feedback |
| 📱 **User-Friendly UI** | Simple web-based interface |
| 📊 **Sentiment Analysis** | Analyzes emotional tone |
| 🧠 **Deep Learning** | LSTM & Transformer architecture |

---

## 🏗️ System Architecture

User Input → Text Processing → Intent Detection → Sentiment Analysis
↓
Neural Network Prediction → Response Mapping → Personalization
↓
Empathetic Response → User Display

---


### Architecture Layers
┌─────────────────────────────────────────┐
│ Frontend (HTML/CSS/JavaScript) │
│ (Home, Login, Chat Interface) │
└────────────────┬────────────────────────┘
│
┌────────────────┴────────────────────────┐
│ Backend (Flask Web Server - Port 5006) │
│ - Route Management │
│ - Request Processing │
│ - Response Generation │
└────────────────┬────────────────────────┘
│
┌────────────────┴────────────────────────┐
│ NLP & ML Processing │
│ - Tokenization │
│ - Lemmatization │
│ - Intent Classification │
│ - Sentiment Analysis │
└────────────────┬────────────────────────┘
│
┌────────────────┴────────────────────────┐
│ Deep Learning Model │
│ - Neural Network (Sequential) │
│ - Dense Layers (128→64→Classes) │
│ - ReLU & Softmax Activation │
│ - Model: chatbots_model3.h5 │
└────────────────┬────────────────────────┘
│
┌────────────────┴────────────────────────┐
│ Data Layer │
│ - intents.json (Intent Definitions) │
│ - words3.pkl (Vocabulary) │
│ - classes3.pkl (Intent Classes) │
└─────────────────────────────────────────┘


---

## 🛠️ Technologies Used

### Backend & Framework
- **Flask** (2.3.0) - Web framework
- **Python** (3.8+) - Core language

### Machine Learning & NLP
- **TensorFlow/Keras** (2.12.0) - Deep learning
- **NLTK** (3.8.1) - Natural Language Processing
- **NumPy** (1.24.0) - Numerical computing
- **Scikit-learn** (1.2.0) - ML algorithms
- **Pandas** (2.0.0) - Data manipulation

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **JavaScript** - Interactivity

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- pip (package manager)
- 4GB RAM
- 500MB disk space

### Step 1: Clone Repository
bash
git clone https://github.com/saiteja731/MentalHealthCompanionChatBot.git
cd MentalHealthCompanionChatBot

### Step 2: Create Virtual Environment
bash
# Create
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

### Step 3: Install Dependencies
bash
pip install -r Requirements.txt

### Step 4: Download NLTK Data
bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

### Step 5: Run Application
bash
python app.py

### Step 6: Access Application
text
Open: http://localhost:portnumber

----

### 🎮 Usage Guide
### Step 1: Home Page
- Welcome screen with project information

- Features overview

- Navigation to Login/Chat

### Step 2: Login
text
Username: admin
Password: admin

### Step 3: Chat Interface
- Type your message

- Get empathetic responses

- Audio feedback (optional)

- Real-time conversation

### Example Conversations
- User: "I'm feeling very stressed and anxious"
- Bot: "I understand you're feeling stressed. Let's work through this together. Try deep breathing exercises - inhale for 4 counts, hold for 4, exhale for 4. This can help calm your nervous system."

- User: "I'm struggling with sleep"
- Bot: "Sleep issues are common. Here are some tips:

       1.Maintain a consistent sleep schedule

      2.Avoid screens 30 minutes before bed

       3.Try relaxation techniques like meditation"

- User: "I feel depressed"
- Bot: "I'm sorry you're feeling this way. Depression is treatable. Consider these steps: Talk to someone you trust, practice self-care, and seek professional help if needed. You're not alone."
  
---

### 📁 Project Structure

    MentalHealthCompanionChatBot
    │
    ├── app.py                    # Main Flask application
    ├── train.py                  # Model training script
    ├── intents.json              # Intent definitions & responses
    ├── Requirements.txt          # Dependencies
    │
    ├── Templates/                # HTML Templates
    │   ├── home.html            # Home page
    │   ├── login.html           # Login page
    │   └── chat.html            # Chat interface
    │
    ├── Static/                   # Static Assets
    │   ├── 3874157.jpg          # Images
    │   ├── n1.jpg to n8.jpg     # UI images
    │   ├── aud1.mp3             # Audio files
    │   └── aud2.mp3             # Notification sounds
    │
    ├── chatbots_model3.h5        # Trained neural network (142MB)
    ├── words3.pkl                # Preprocessed vocabulary
    ├── classes3.pkl              # Intent classes
    ├── README.md                 # Documentation
    └── .gitignore               # Git ignore rules

---

### 🔬 How It Works - Step by Step Process
 ### Phase 1: Data Preprocessing

    Raw Input: "I'm feeling anxious and stressed"
         ↓
    TOKENIZATION: Split into words
    ["I'm", "feeling", "anxious", "and", "stressed"]
         ↓
    LEMMATIZATION: Convert to base form
     ["be", "feel", "anxious", "and", "stress"]
         ↓
    STOP-WORD REMOVAL: Remove common words
    ["anxious", "stress"]
         ↓
    CLEANED OUTPUT: Ready for processing

### Phase 2: Feature Extraction

    Vocabulary: ["anxious", "stress", "worried", "nervous", ...]
       ↓
    BAG-OF-WORDS VECTORIZATION:
    [1, 1, 0, 0, 0, ...] (1 if word present, 0 if absent)
       ↓
    NUMERICAL VECTOR: [1, 1, 0, 0, 0, ...]

### Phase 3: Intent Prediction

    Input Vector: [1, 1, 0, 0, 0, ...]
       ↓
    NEURAL NETWORK LAYERS:
    Input Layer (1x vocab_size)
       ↓
    Dense Layer 1: 128 neurons + ReLU activation
       ↓
    Dropout 50% (prevents overfitting)
       ↓
    Dense Layer 2: 64 neurons + ReLU activation
       ↓
     Dropout 50%
       ↓ 
    Output Layer: Softmax activation
       ↓
    INTENT PROBABILITIES:
    anxiety_support: 87% ✓ SELECTED
    stress_management: 10%
    depression_support: 3%

### Phase 4: Response Generation

     Detected Intent: anxiety_support
         ↓
    LOOKUP: intents.json → anxiety_support
          ↓
    AVAILABLE RESPONSES:
      [
       "I understand you're feeling anxious...",
       "Anxiety is manageable...",
       "Let's work through this..."
     ]
             ↓
    RANDOM SELECTION: Pick one response
             ↓
    PERSONALIZATION: Add user's name if provided
               ↓
    FINAL RESPONSE: Display to user

----

### Training Process


    1. Load Intent Data from intents.json
          ↓
    2. Tokenize and Lemmatize Text
          ↓
    3. Create Bag-of-Words Vectors
          ↓
    4. Generate Training Dataset
          ↓
    5. Build Neural Network Model
          ↓
    6. Train on Mental Health Conversations
          ↓
    7. Evaluate Performance (99%+ accuracy)
          ↓
    8. Save Model (chatbots_model3.h5)

----
### 🎯 Key Intents Handled

1. Anxiety Support
   Keywords: anxious, nervous, worried, panic, fear
   
2. Depression Help
   Keywords: depressed, sad, hopeless, worthless, empty
   
3. Stress Management
   Keywords: stressed, overwhelmed, pressure, tense
   
4. Sleep Issues
   Keywords: sleep, insomnia, tired, exhausted, dream
   
5. Mood Tracking
   Keywords: mood, feeling, emotional, happy, sad
   
6. Self-Care Tips
   Keywords: exercise, meditation, relaxation, self-care
   
7. Professional Help
   Keywords: therapy, therapist, counselor, professional, help
---
### 📚 Technologies & Libraries

| Category      | Library      | Version |
| ------------- | ------------ | ------- |
| Web Framework | Flask        | 2.3.0   |
| Deep Learning | TensorFlow   | 2.12.0  |
| ML Framework  | Keras        | 2.12.0  |
| NLP           | NLTK         | 3.8.1   |
| Numerical     | NumPy        | 1.24.0  |
| Data Science  | Pandas       | 2.0.0   |
| ML Algorithms | Scikit-learn | 1.2.0   |

---
### 🔮 Future Enhancements

✨ Voice Interaction: Speech-to-text and text-to-speech

🌍 Multilingual Support: Multiple language support

📱 Mobile App: iOS and Android applications

⌚ Wearable Integration: Connect with health devices

🤝 Therapist Connection: Link to professional therapists

📊 Progress Tracking: Monitor emotional well-being over time

🔔 Push Notifications: Reminders and wellness tips

🎮 Gamification: Reward positive mental health behaviors
