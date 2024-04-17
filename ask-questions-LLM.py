import gradio as gr

def load_file(file_info):
    # Read the uploaded file.
    with open(file_info['name'], 'r') as f:
        content = f.read()
    return content

def display_content(content):
    return content

def ask_questions(content):
    # Example logic to parse questions and provide automated answers.
    questions = [line for line in content.split('\n') if line.startswith('Q: ')]
    answers = ['A: Yes' if i % 2 == 0 else 'A: No' for i, _ in enumerate(questions)]
    return '\n'.join([f"{q} - {a}" for q, a in zip(questions, answers)])

def download_results(content):
    # Return the content for downloading. Ensure it is returned as bytes.
    return content.encode('utf-8')

iface = gr.Blocks()

with iface:
    with gr.Row():
        file_uploader = gr.FileUploader(label='Upload a txt file')
        submit_btn = gr.Button("Load File")
        
    with gr.Column():
        content_display = gr.Textbox(label="Content of the uploaded text file", interactive=False)
        submit_btn.click(load_file, inputs=file_uploader, outputs=content_display)
    
    ask_btn = gr.Button("Ask Questions")
    questions_display = gr.Textbox(label='Questions and Answers', interactive=False)
    ask_btn.click(ask_questions, inputs=content_display, outputs=questions_display)
    
    download_btn = gr.Button("Download Results")
    gr.InterfaceS3Download(download_results, source=questions_display, download_button=download_btn)

iface.launch()
