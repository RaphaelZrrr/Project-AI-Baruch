import gradio as gr  # Import the Gradio library
from transformers import pipeline  # Import the pipeline function from the transformers library

# Load the summarization pipeline with the chosen model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to split the text into chunks
def split_text(text, max_chunk_size=1024):
    sentences = text.split('. ')  # Split the text into sentences
    chunks = []  # Initialize an empty list to hold the chunks
    current_chunk = ""  # Start with an empty chunk
    for sentence in sentences:
        # If adding the sentence keeps the chunk size within the limit
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + ". "  # Add the sentence to the current chunk
        else:
            chunks.append(current_chunk.strip())  # Save the chunk and start a new one
            current_chunk = sentence + ". "  # Add the sentence to the new chunk
    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append(current_chunk.strip())
    return chunks  # Return the list of chunks

# Function to summarize the text
def summarize_text(input_text):
    chunks = split_text(input_text)  # Split the input text into chunks
    # Summarize each chunk and collect the summaries
    summaries = [summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text'] for chunk in chunks]
    final_summary = " ".join(summaries)  # Combine all summaries into one text
    return final_summary  # Return the final summary

# Enhanced Gradio interface
with gr.Blocks(css="""
    body { 
        background: linear-gradient(to right, #4facfe, #00f2fe); 
        font-family: Arial, sans-serif; 
        color: #333333;
    }
    .input-container, .output-container {
        padding: 20px;
        border: 1px solid #cccccc;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    #summarize-btn {
        background-color: #007BFF; /* Change the color to blue */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    #summarize-btn:hover {
        background-color: #0056b3; /* Darker blue for hover */
        transform: scale(1.05);
    }
    h1, .centered-text {
        text-align: center; /* Center the title and the text */
    }
""") as demo:  # Create a Gradio Blocks interface with custom CSS

    gr.Markdown("# Text Summarization App")  # Add a centered title
    gr.Markdown(
        "Enter a paragraph below, and this app will generate a concise summary using Hugging Face's BART model.", 
        elem_classes=["centered-text"]  # Center this description text
    )

    # Visual containers
    with gr.Row(elem_classes=["input-container"]):  # Create a container for the input
        text_input = gr.Textbox(
            label="Input Text",  # Label for the input box
            lines=10,  # Number of visible lines
            placeholder="Type or paste a long paragraph here..."  # Placeholder text
        )
    with gr.Row(elem_classes=["output-container"]):  # Create a container for the output
        summary_output = gr.Textbox(label="Summary", lines=5, interactive=False)  # Output box for the summary

    # Submission button
    summarize_button = gr.Button("Summarize", elem_id="summarize-btn")  # Add a button with custom styling
    summarize_button.click(summarize_text, inputs=[text_input], outputs=[summary_output])  # Link the button to the summarize_text function

    # Help section
    gr.Markdown("""
    ### How to Use
    1. Enter a paragraph in the "Input Text" box.
    2. Click on "Summarize" to generate a concise summary.
    3. See the result in the "Summary" box.
    """)

demo.launch(share=True)  # Launch the Gradio app and allow sharing
