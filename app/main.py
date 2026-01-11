import os
import gradio as gr
from dotenv import load_dotenv
from app.chatbot import CareerAdvisorChatbot

# Load environment variables (from .env file)
load_dotenv()


def main():
    # Initialize the chatbot
    # Note: DB connections are handled inside the class based on env vars
    chatbot = CareerAdvisorChatbot()

    def chat_fn(user_query: str, history: list):
        """
        Main function called by Gradio for every message.
        """
        # 1. Reset context if this is a new conversation
        if not history:
            chatbot.reset()

        # 2. Get response from the agent
        result = chatbot.chat(user_query)

        final_response = result.response
        tool_calls = result.tool_calls

        # 3. Append Tool/Debug Info (similar to your old project)
        if tool_calls:
            final_response += "\n\n--- 🛠 **[Developer Info] Agent used these tools** ---\n"
            for call in tool_calls:
                tool_name = call.get('tool')
                args = call.get('args')
                tool_output = str(call.get('result'))

                # Truncate long database outputs for cleaner UI
                if len(tool_output) > 300:
                    tool_output = tool_output[:300] + "... [truncated]"

                final_response += f"**🔹 Tool:** `{tool_name}`\n"
                final_response += f"**Arguments:** `{args}`\n"
                final_response += f"**DB Output:** _{tool_output}_\n\n"

        return final_response

    def authenticate(username, password):
        """
        Simple auth using env vars. Defaults to admin/admin if not set.
        """
        valid_username = os.environ.get("GRADIO_USERNAME", "admin")
        valid_password = os.environ.get("GRADIO_PASSWORD", "admin")
        return username == valid_username and password == valid_password

    # 4. Configure the UI
    chatbot_ui = gr.ChatInterface(
        fn=chat_fn,
        title="🎓 Academic Career Advisor",
        description="Ask about career paths (e.g., 'Data Scientist') to find relevant school courses, skills, and research papers.",
        examples=[
            "How do I become a Data Scientist?",
            "What courses teach Machine Learning?",
            "Find research papers about Artificial Intelligence."
        ],
        textbox=gr.Textbox(placeholder="Ask a question...", container=False, scale=7),
        fill_height=True,
    )

    # 5. Launch
    print("Starting Gradio Server on port 7860...")
    print("Chatbot will be accessable at http://localhost:7860")
    chatbot_ui.launch(
        server_name="0.0.0.0",  # Required for Docker mapping
        server_port=7860,
        auth=authenticate,
        show_error=True,
        share=False  # Set to True if you want a public gradio link
    )



if __name__ == "__main__":
    main()