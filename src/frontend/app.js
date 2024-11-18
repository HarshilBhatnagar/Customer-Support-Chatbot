
function displayMessage(message, sender) {
    const chatbox = document.getElementById("chatbox");
    const messageElement = document.createElement("div");
    messageElement.classList.add("message");

    if (sender === "User") {
        messageElement.classList.add("user-message");
    } else {
        messageElement.classList.add("bot-message");
    }

    messageElement.textContent = message;
    chatbox.appendChild(messageElement);
    chatbox.scrollTop = chatbox.scrollHeight;
}

async function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    if (!userInput) return;

    displayMessage(userInput, "User");

    document.getElementById("userInput").value = "";

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: userInput })
        });

        const data = await response.json();

        if (data.response) {
            displayMessage(data.response, "Bot");
        } else {
            displayMessage("Sorry, I didn't understand that.", "Bot");
        }
    } catch (error) {
        console.error("Error:", error);
        displayMessage("An error occurred. Please try again later.", "Bot");
    }
}
