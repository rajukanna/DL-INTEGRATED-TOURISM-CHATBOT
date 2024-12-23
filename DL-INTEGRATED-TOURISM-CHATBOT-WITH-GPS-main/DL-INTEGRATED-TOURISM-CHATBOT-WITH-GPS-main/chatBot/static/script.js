
const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");

let userLocation = null;

const createChatLi = (message, className) => {
    console.log("Creating chat list item"); // Debugging log
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi;
}

const displayMessage = (message, type) => {
    console.log(`Displaying message: ${message}`); // Debugging log
    const className = type === "incoming" ? "incoming" : "outgoing";
    const chatLi = createChatLi(message, className);
    chatbox.appendChild(chatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);
}

const sendMessageToBackend = (message) => {
    console.log(`Sending message to backend: ${message}`); // Debugging log
    const payload = { message: message };
    if (userLocation) {
        payload.userLocation = userLocation;
    }

    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        displayMessage(data.response, "incoming");
    })
    .catch(error => {
        console.error("Error sending message:", error);
        displayMessage("Oops! Something went wrong.", "incoming");
    });
}

const handleChat = () => {
    console.log("Handling chat"); // Debugging log
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;

    chatInput.value = ""; // Clear the input field

    // Display user's message in the chatbox
    displayMessage(userMessage, "outgoing");

    // Send user's message to the backend and get response
    sendMessageToBackend(userMessage);
}

sendChatBtn.addEventListener("click", handleChat);
chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        handleChat();
    }
});

closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));

// Check if geolocation is supported
if ("geolocation" in navigator) {
    console.log("Geolocation is supported"); // Debugging log
    // Ask for permission to access location
    navigator.geolocation.getCurrentPosition(
        // Success callback
        function(position) {
            // Handle successful retrieval of location
            console.log("Geolocation success"); // Debugging log
            userLocation = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude
            };
            displayMessage(`Your current location is approximately at latitude ${userLocation.latitude} and longitude ${userLocation.longitude}.`, "incoming");
        },
        // Error callback
        function(error) {
            console.error("Error getting location:", error.message); // Debugging log
            // Handle errors
            switch(error.code) {
                case error.PERMISSION_DENIED:
                    console.log("User denied the request for Geolocation.");
                    break;
                case error.POSITION_UNAVAILABLE:
                    console.log("Location information is unavailable.");
                    break;
                case error.TIMEOUT:
                    console.log("The request to get user location timed out.");
                    break;
                case error.UNKNOWN_ERROR:
                    console.log("An unknown error occurred.");
                    break;
            }
            // Inform the user about the error in accessing their location
            displayMessage("Error: Unable to access your location.", "incoming");
        }
    );
} else {
    console.log("Geolocation is not supported by this browser.");
    // Inform the user that their browser does not support geolocation
    displayMessage("Error: Geolocation is not supported by this browser.", "incoming");
}
