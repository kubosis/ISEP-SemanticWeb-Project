import os
import gradio as gr
from dotenv import load_dotenv
from app.chatbot import CareerAdvisorChatbot

load_dotenv()


def main():
    chatbot = CareerAdvisorChatbot()

    def chat_fn(user_query: str, history: list):
        """
        Main function called by Gradio for every message.
        """
        if not history:
            chatbot.reset()

        result = chatbot.chat(user_query)

        final_response = result.response
        tool_calls = result.tool_calls

        # Append Tool/Debug Info
        if tool_calls and os.getenv("ENV", "prod").lower() == "dev":
            final_response += "\n\n--- 🛠 **[Developer Info] Agent used these tools** ---\n"
            for call in tool_calls:
                tool_name = call.get('tool')
                args = call.get('args')
                tool_output = str(call.get('result'))

                if len(tool_output) > 300:
                    tool_output = tool_output[:300] + "... [truncated]"

                final_response += f"**🔹 Tool:** `{tool_name}`\n"
                final_response += f"**Arguments:** `{args}`\n"
                final_response += f"**DB Output:** _{tool_output}_\n\n"

        return final_response

    def authenticate(username, password):
        valid_username = os.environ.get("GRADIO_USERNAME",)
        valid_password = os.environ.get("GRADIO_PASSWORD",)
        return username == valid_username and password == valid_password

    # Custom CSS to force full screen
    custom_css = """
    /* Remove default margins and padding */
    body, html {
        margin: 0;
        padding: 0;
        height: 100vh;
        overflow: hidden;
    }

    /* Make Gradio container fill viewport */
    .gradio-container {
        max-width: 100% !important;
        width: 100vw !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Make main block fill height */
    .main {
        height: 100vh !important;
    }

    /* Adjust chat container */
    .contain {
        max-width: 100% !important;
    }

    /* Make chatbot fill available space */
    #component-0 {
        height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Ensure chat messages area expands */
    .chatbot {
        flex-grow: 1 !important;
        height: 100% !important;
    }
    """

    with gr.Blocks(
            fill_height=True,
            css=custom_css
    ) as demo:

        gr.ChatInterface(
            fn=chat_fn,
            title="🎓 Academic Career Advisor",
            description="Ask about career paths to find relevant courses and papers.",
            examples=[
                "How do I become a Data Scientist?",
                "What courses teach Machine Learning?",
                "Find research papers about Artificial Intelligence."
            ],
            textbox=gr.Textbox(
                placeholder="Ask a question...",
                container=False,
                scale=7
            ),
            fill_height=True,
            chatbot=gr.Chatbot(height=700),  # Set minimum height
        )

    print("Starting Gradio Server on port 7860...")
    print("Chatbot will be accessible at http://localhost:7860")
    share_enabled = os.getenv("GRADIO_SHARE", "False").lower() == "true"
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=authenticate,
        show_error=True,
        theme = gr.themes.Soft(),
        share = share_enabled
    )


if __name__ == "__main__":
    main()