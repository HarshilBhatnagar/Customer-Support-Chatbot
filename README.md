<h1>Customer Support Chatbot 🚀</h1>

<p>An AI-powered chatbot designed to handle customer queries efficiently, built using state-of-the-art machine learning models for intent recognition, named entity recognition, and dynamic dialogue generation.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#models">Models</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#project-structure">Project Structure</a></li>
  <li><a href="#limitations">Limitations</a></li>
  <li><a href="#future-enhancements">Future Enhancements</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>The Customer Support Chatbot leverages advanced AI techniques to understand user queries, extract key entities, and generate meaningful responses. This chatbot aims to enhance user experience by automating common customer service tasks such as checking order status, handling cancellations, and providing product information.</p>

<p><strong>Note:</strong> Due to GitHub’s file size limitations, the trained model files are not included in the repository. However, you can generate the models locally by running the provided training scripts.</p>

<h2 id="features">Features</h2>
<ul>
  <li><strong>Intent Recognition:</strong> Accurately classifies user intents using a fine-tuned BERT model.</li>
  <li><strong>Named Entity Recognition (NER):</strong> Identifies key entities such as order IDs using a custom-trained NER model.</li>
  <li><strong>Dynamic Dialogue Generation:</strong> Uses a fine-tuned DialoGPT model for generating context-aware responses.</li>
  <li><strong>Seamless API Integration:</strong> Built with a Flask backend and a user-friendly frontend interface.</li>
</ul>

<h2 id="dataset">Dataset</h2>
<p>The chatbot was trained using a diverse, augmented dataset containing over 3,700 examples of customer queries and responses:</p>
<ul>
  <li><strong>Conversation Data:</strong> Includes multi-turn dialogues for realistic conversation flow.</li>
  <li><strong>FAQ Data:</strong> Contains frequently asked questions and their corresponding answers.</li>
  <li><strong>NER Data:</strong> Annotated data for training the NER model to recognize entities like order IDs.</li>
</ul>

<h2 id="models">Models</h2>
<p>The project includes the following models:</p>
<ul>
  <li><strong>Intent Recognition Model:</strong> Built using BERT for accurate intent classification.</li>
  <li><strong>NER Model:</strong> Custom-trained model for extracting named entities from user input.</li>
  <li><strong>Dialogue Generation Model:</strong> Fine-tuned DialoGPT model for generating dynamic responses.</li>
</ul>
<p><strong>Note:</strong> Due to the large file size, the <code>models/</code> folder could not be uploaded to the repository. You can train the models locally using the provided training scripts or load pretrained models using the scripts in the <code>src/</code> directory.</p>

<h2 id="installation">Installation</h2>
<p>To run this project locally, follow these steps:</p>

<h3>Clone the Repository:</h3>
<pre><code>git clone https://github.com/HarshilBhatnagar/Customer-Support-Chatbot.git
cd Customer-Support-Chatbot
</code></pre>

<h3>Install Dependencies:</h3>
<p>Make sure you have Python 3.8 or higher installed. Then, run:</p>
<pre><code>pip install -r requirements.txt
</code></pre>

<h3>Download or Train the Models:</h3>
<p>Run the model training scripts in the <code>src/</code> folder to generate the required models:</p>
<pre><code>python src/train_intent_model.py
python src/train_entity_model.py
python src/train_dialogue_model.py
</code></pre>

<h2 id="usage">Usage</h2>

<h3>Start the Backend Server:</h3>
<pre><code>python src/backend.py
</code></pre>

<h3>Start the Frontend:</h3>
<p>Navigate to the <code>frontend</code> directory and start a local server:</p>
<pre><code>cd frontend
python -m http.server 8000
</code></pre>

<h3>Access the Chatbot Interface:</h3>
<p>Open your browser and go to:</p>
<pre><code>http://127.0.0.1:8000/index.html
</code></pre>

<h2 id="project-structure">Project Structure</h2>
<pre><code>
Customer-Support-Chatbot/
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Preprocessed datasets
├── models/                    # (Not included due to large file size)
├── src/
│   ├── backend.py             # Flask backend API
│   ├── train_intent_model.py  # Intent model training script
│   ├── train_entity_model.py  # NER model training script
│   ├── train_dialogue_model.py# Dialogue model training script
├── frontend/                  # Frontend web interface
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
</code></pre>

<h2 id="limitations">Limitations</h2>
<ul>
  <li><strong>File Size Restrictions:</strong> The pretrained model files are too large to include in the repository. Please use the training scripts to generate the models locally.</li>
  <li><strong>Limited Responses:</strong> The chatbot may echo user input if the query falls outside the scope of the trained dataset.</li>
</ul>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.</p>

<h2>Chatbot Interface</h2>
<p><img src="https://github.com/user-attachments/assets/499a4bbb-7eac-4892-864f-efce1f383289" alt="Chatbot interface"></p>
