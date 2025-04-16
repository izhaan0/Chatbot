class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value.trim(); // Trim to remove extra spaces

        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);
        textField.value = '';  // Clear input field immediately

        // fetch('http://127.0.0.1:5000/predict', {
        //     method: 'POST',
        //     body: JSON.stringify({ messages: text1 }), // Corrected key name
        //     mode: 'cors',
        //     headers: { 'Content-Type': 'application/json' }
        // })
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: text1 })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Server Response:", data);  // Debugging response

            if (data.answer) {
                let msg2 = { name: "Sam", message: data.answer };
                this.messages.push(msg2);
            } else {
                console.error("Invalid response:", data);
                let msg2 = { name: "Sam", message: "Error: No response from server" };
                this.messages.push(msg2);
            }

            this.updateChatText(chatbox);
        })
        .catch(error => {
            console.error("Fetch error:", error);
            let msg2 = { name: "Sam", message: "Error: Failed to connect to server" };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item) {
            if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();
