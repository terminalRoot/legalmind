

# LegalMind

## Description
LegalMind provides off-the-shelf finetuned LLM for legal laws and acts. We have finetuned meta-llama/Llama-2-7b-hf available off the shelf at huggingface. This model is only trained in English language. We want to furthere train for multiple languages as well as global laws.


## Contributing
We welcome any and all contributions! Here are some ways you can get started:
1. Report bugs: If you encounter any bugs, please let us know. Open up an issue and let us know the problem.
2. Contribute code: If you are an AI developer and want to contribute, follow the instructions below to get started!
3. Contribute dataset: If you are AI enthusiast and want to contribute, please share additional dataset and 
3. Suggestions: If you don't want to code but have some awesome ideas, open up an issue explaining some updates or imporvements you would like to see!
4. Documentation: If you see the need for some additional documentation, feel free to add some!

## Setup instruction
1. We use QLoRA, PEFT, and 4Bit quantization to load and train the model on limited hardware.
2. Get request access for Llama2 from [Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
3. Get the [huggingface API](https://huggingface.co/settings/tokens) token to download the model.
4. Set the enviornment for huggingface in fine-tune.py file.
5. Keep the dataset in data folder as data.csv with single column header as text.
6. Run the file with specific configuration:
    ```
    python fine-tune.py --model_name meta-llama/Llama-2-7b-hf \
     --load_in_4bit \
     --use_peft \
     --batch_size 4 \
     --gradient_accumulation_steps 2 \
     --num_train_epochs 5 \
     --output_dir output-paragraph
     ```
7. We are using Intel Accelerated APIs to further run the inference model on CPU only hardware.
8. Run the inference after training completes:
    ```python inference.py```

## Instructions
1. Fork this repository
2. Clone the forked repository
3. Add your contributions (code or documentation)
4. Commit and push
5. Wait for pull request to be merged


