import os
import sys
import gradio as gr
from dotenv import load_dotenv
from app.chatbot import CareerAdvisorChatbot

load_dotenv()

chatbot = CareerAdvisorChatbot()

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


def chat_fn(user_query: str, history: list):
    if not history:
        chatbot.reset()

    result = chatbot.chat(user_query)
    final_response = result.response
    tool_calls = result.tool_calls

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
    valid_username = os.environ.get("GRADIO_USERNAME", "admin")
    valid_password = os.environ.get("GRADIO_PASSWORD", "admin")
    return username == valid_username and password == valid_password


def main():
    # Check share setting
    share_enabled = os.getenv("GRADIO_SHARE", "False").lower() == "true"

    print("=" * 60)
    print("Starting Academic Career Advisor Chatbot")
    print("=" * 60)
    print(f"Share Mode: {'ENABLED' if share_enabled else 'DISABLED'}")
    print(f"Cache Directory: {os.getenv('XDG_CACHE_HOME', 'default')}")
    print(f"Server: http://0.0.0.0:7860")
    print("=" * 60)
    sys.stdout.flush()

    # Custom CSS for full screen

    with gr.Blocks(
            fill_height=True,
            title="Academic Career Advisor",
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
            chatbot=gr.Chatbot(height=700),
        )

    try:
        print("Launching Gradio interface...")
        sys.stdout.flush()

        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            auth=authenticate,
            show_error=True,
            share=share_enabled,
            debug=True,
            quiet=False,
            css=custom_css,
            inbrowser=False,
        )
    except Exception as e:
        print(f"Error launching Gradio: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    main()