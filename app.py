import gradio as gr
from src.model_byt5.tokenizer import Tokenizer_byt5
from src.model_byt5.model import Transformer_byt5
import json
import torch


def translate(model_path, inputs):
    model_weights_path = f"{model_path}/pytorch_model.bin"
    model_config_path = f"{model_path}/config.json"
    with open(model_config_path, 'r') as f:
        config = json.load(f)

    state_dict = torch.load(model_weights_path)
    model = Transformer_byt5(config=config)
    model.load_state_dict(state_dict)
    model = model.eval()
    tokenizer = Tokenizer_byt5()  
    ids = tokenizer(inputs, max_length=512)
    len_pad = 512 - len(ids)
    if len_pad > 0:
        ids = ids + [0 for x in range(len_pad)]
    print(ids)  
    inputs = torch.tensor([ids]).to(torch.device('cpu'))
    outputs = model.generate(inputs, max_length=512)
    text = tokenizer.ids2text(outputs.tolist()[0])
    return text

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.components.Textbox(label="checkpoint path, eg: ./mt5-finetuned/checkpoint-34000", value="./mt5-finetuned/checkpoint-34000"),
        gr.components.Textbox(label="INPUTS"),
    ],
    outputs=["text"],
    cache_examples=False,
    title="Test",
    description="Test"
)

demo.launch(debug=True, share=True, server_name="0.0.0.0")
