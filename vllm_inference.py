import pandas as pd
import numpy as np

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def model_def(model_path,no_gpus):
    model = LLM(model_path,
                tensor_parallel_size=no_gpus,
                max_model_len=20000,
                gpu_memory_utilization = 0.97,
                enforce_eager=True,
                enable_chunked_prefill=True,
                download_dir=model_path,)

    sampling_params = SamplingParams(temperature=0.6,top_p=0.9,max_tokens=2500) 

    return model, sampling_params




def prediction(file,model_path,no_gpus,tokenizer):
    df=pd.read_csv(file).iloc[:10]
    
    templated_prompts = []
    for text in df['Text']:

        prompt=[{"role": "user", "content": text}]
        templated_prompt  = tokenizer.apply_chat_template(prompt,add_generation_prompt = True,tokenize=False)   
        templated_prompts.append(templated_prompt)

        
    print(templated_prompts[0]) 

    model, sampling_params= model_def(model_path=model_path,no_gpus=no_gpus)
    result = model.generate(templated_prompts, sampling_params)
    
    output=[(x.outputs[0].text) for x in result]
    # print([i+'\n\n*****' for i in output])

    df['prediction']=output
    df['prompt']=templated_prompts
   

    return df


if __name__ == "__main__":
    model_path='DeepSeek-R1-Distill-Llama-70B'
    tokenizer_path= model_path
    download_dir=model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

