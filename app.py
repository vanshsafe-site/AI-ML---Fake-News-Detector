import gradio as gr
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)

    return "🟢 REAL NEWS" if pred[0] == 1 else "🔴 FAKE NEWS"

with gr.Blocks() as demo:
    gr.Markdown("# 📰 Fake News Detector AI")
    gr.Markdown("Detect whether a news article is real or fake using Machine Learning")

    input_box = gr.Textbox(label="Enter News Text", lines=5)
    output_box = gr.Textbox(label="Result")

    btn = gr.Button("Check News")

    btn.click(predict, input_box, output_box)

    gr.Examples(
        examples=[
            "Government announces new education policy for students",
            "Aliens landed in India and met government officials"
        ],
        inputs=input_box
    )

demo.launch()