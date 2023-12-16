import os
os.environ['TRANSFORMERS_CACHE'] = './huggingface'
os.environ['HF_HOME'] = './huggingface'

from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
import intel_extension_for_pytorch as ipex
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import gradio as gr 
from PIL import Image
import pytesseract

is_img = False

def image_to_text(image_data):
    global is_img
    try:
        image = Image.fromarray(image_data.astype('uint8'), 'RGB')
        text = pytesseract.image_to_string(image, lang='eng+hin')
        is_img = True
        return text
    except:
        is_img = False
        return ''

def chat_with_llm(chat_history, user_input):
    global is_img
    
    print(f'user input: {user_input} chat history: {chat_history}')
    print(f'is_img: {is_img}\n')

    if is_img:
        bot_response = local_llm(user_input)
        
        is_img = False
    else:
        bot_response = qa_chain({"query": user_input})
        bot_response = bot_response['result']

    print(bot_response)
    response = ""
    for letter in ''.join(bot_response):
        response += letter + ""
        yield chat_history + [(user_input, response)]

checkpoint = "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

bnb_config = BitsAndBytesConfig(
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,#float16
    )

base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-12-v3")
db = Chroma(persist_directory="vector_data", embedding_function=embeddings)

pipe = pipeline(
    'text2text-generation',
    model = base_model,
    tokenizer = tokenizer,
    max_length = 2048,
    do_sample = True,
    temperature = 0.3,
    top_p= 0.95,
    repetition_penalty= 1.2,
)

device = "xpu"
local_llm = HuggingFacePipeline(pipeline=pipe)

try:
    ipex.optimize_transformers(local_llm, dtype=torch_dtype)
except:
    ipex.optimize(local_llm, dtype=torch_dtype)

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":2}),
        return_source_documents=True,
        )

with gr.Blocks() as gradioUI:
    with gr.Row():
        chatbot = gr.Chatbot()

    with gr.Row():
        input_query = gr.TextArea(label='Input', placeholder='Type here or use OCR text...', show_copy_button=True)
        image_upload = gr.Image(label='Upload Image')

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_input_btn = gr.Button("Clear Input")
        clear_chat_btn = gr.Button("Clear Chat")
    
    image_upload.change(image_to_text, inputs=image_upload, outputs=input_query)
    
    submit_btn.click(chat_with_llm, inputs=[chatbot, input_query], outputs=chatbot)
    clear_input_btn.click(lambda: "", inputs=None, outputs=input_query)
    clear_chat_btn.click(lambda: [], inputs=None, outputs=chatbot)

gradioUI.launch()
