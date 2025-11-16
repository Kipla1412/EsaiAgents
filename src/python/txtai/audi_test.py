# from txtai.pipeline import Microphone, Transcription
# microphone = Microphone()
# transcriber = Transcription("openai/whisper-tiny.en")

# print("Listening for speech. Speak now...")

# audio_data, rate = microphone()
# text = transcriber((audio_data, rate))

# print(f"You said: {text}")
import gradio as gr
from txtai.agent.check.medgemma import medgemma_tool

def medgemma_chat(text, image):
    if image is None:
        return medgemma_tool(text) 
    else:
        return medgemma_tool(text, image_path=image)

with gr.Blocks() as demo:
    gr.Markdown("# 🧑‍⚕️ MedGemma Chat (Text + Image)")

    text = gr.Textbox(label="Enter your question")
    image = gr.Image(type="filepath", label="Upload an image (optional)")
    output = gr.Textbox(label="MedGemma Response")

    gr.Button("Ask").click(fn=medgemma_chat, inputs=[text, image], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
