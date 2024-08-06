## Updated:  
    1. fixed AMRBART.yaml  

## Preprocessing:  
    1. install AMRBART.yaml into your env  
    2. change the root name in your side  

[change this line](https://github.com/tingchihc/claim_generation/blob/fbd7df37f8e63edf499a6122467b7e736981bb7d/AMRBART/fine-tune/main.py#L277)  

## Excute:  
    python claim_generation.py --input test/graph_no_quotes.json --output test/claim.json  

## Reference  
    https://github.com/goodbai-nlp/AMRBART  
    https://github.com/goodbai-nlp/AMR-Process  
