<h1>Customer Support Chatbot 🚀<h1/>


An advanced AI-powered chatbot designed to handle customer queries efficiently, built using state-of-the-art machine learning models for intent recognition, named entity recognition, and dynamic dialogue generation.


Table of Contents
Overview
Features
Dataset
Models
Installation
Usage
Project Structure
Limitations
Future Enhancements
Contributing
License


Overview
The Customer Support Chatbot leverages advanced AI techniques to understand user queries, extract key entities, and generate meaningful responses. This chatbot aims to enhance user experience by automating common customer service tasks such as checking order status, handling cancellations, and providing product information.

Note: Due to GitHub’s file size limitations, the trained model files are not included in the repository. However, you can generate the models locally by running the provided training scripts.


Features
Intent Recognition: Accurately classifies user intents using a fine-tuned BERT model.
Named Entity Recognition (NER): Identifies key entities such as order IDs using a custom-trained NER model.
Dynamic Dialogue Generation: Uses a fine-tuned DialoGPT model for generating context-aware responses.
Seamless API Integration: Built with a Flask backend and a user-friendly frontend interface.


Dataset
The chatbot was trained using a diverse, augmented dataset containing over 3,700 examples of customer queries and responses.

Conversation Data: Includes multi-turn dialogues for realistic conversation flow.
FAQ Data: Contains frequently asked questions and their corresponding answers.
NER Data: Annotated data for training the NER model to recognize entities like order IDs.
Models
The project includes the following models:

Intent Recognition Model: Built using BERT for accurate intent classification.
NER Model: Custom-trained model for extracting named entities from user input.
Dialogue Generation Model: Fine-tuned DialoGPT model for generating dynamic responses.
Note: Due to the large file size, the models/ folder could not be uploaded to the repository. You can train the models locally using the provided training scripts or load pretrained models using the scripts in the src/ directory.


Installation
To run this project locally, follow these steps:

Clone the Repository:
bash
Copy code
git clone https://github.com/HarshilBhatnagar/Customer-Support-Chatbot.git
cd Customer-Support-Chatbot
Install Dependencies:
Make sure you have Python 3.8 or higher installed. Then, run:

bash
Copy code
pip install -r requirements.txt
Download or Train the Models:
Run the model training scripts in the src/ folder to generate the required models:

bash
Copy code
python src/train_intent_model.py
python src/train_entity_model.py
python src/train_dialogue_model.py
Usage
Start the Backend Server:
bash
Copy code
python src/backend.py
Start the Frontend:
Navigate to the frontend directory and start a local server:

bash
Copy code
cd frontend
python -m http.server 8000
Access the Chatbot Interface:
Open your browser and go to:

arduino
Copy code
http://127.0.0.1:8000/index.html


Limitations
File Size Restrictions: The pretrained model files are too large to include in the repository. Please use the training scripts to generate the models locally.
Limited Responses: The chatbot may echo user input if the query falls outside the scope of the trained dataset.

Chatbot Interface
![Chatbot interface](https://github.com/user-attachments/assets/499a4bbb-7eac-4892-864f-efce1f383289)







