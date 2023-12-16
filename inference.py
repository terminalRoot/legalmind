import os
os.environ['TRANSFORMERS_CACHE'] = './huggingface'
os.environ['HF_HOME'] = './huggingface'

from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig
import torch
import intel_extension_for_pytorch as ipex
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import gradio as gr 

def chat(chat_history, user_input):
    print('user input: %s' % user_input)
    print('chat history: %s' % chat_history)
    
    bot_response = qa_chain({"query": user_input})
    bot_response = bot_response['result']
    response = ""
    for letter in ''.join(bot_response):
        response += letter + ""
        yield chat_history + [(user_input, response)]

checkpoint = "llm_models/output-paragraph"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,#float16
    )


device = "xpu"
base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device)
.eval()

try:
    ipex.optimize_transformers(base_model, dtype=torch_dtype)
except:
    ipex.optimize(base_model, dtype=torch_dtype)

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-12-v3")
db = Chroma(persist_directory="vector_data", embedding_function=embeddings)


pipe = pipeline(
    'text2text-generation',
    model = base_model,
    tokenizer = tokenizer,
    max_length = 512,
    do_sample = True,
    temperature = 0.3,
    top_p= 0.95
)
local_llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":2}),
        return_source_documents=True,
        )



with gr.Blocks() as gradioUI:

    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        input_query = gr.TextArea(label='Input',show_copy_button=True)

    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            clear_input_btn = gr.Button("Clear Input")
        with gr.Column():
            clear_chat_btn = gr.Button("Clear Chat")

    submit_btn.click(chat, [chatbot, input_query], chatbot)
    submit_btn.click(lambda: gr.update(value=""), None, input_query, queue=False)
    clear_input_btn.click(lambda: None, None, input_query, queue=False)
    clear_chat_btn.click(lambda: None, None, chatbot, queue=False)

gradioUI.queue().launch()

