cd src
python -m task.truthful_qa \
-m "your/path/to/llm" \
--nli "your/path/to/deberta-v3-large-mnli" \
--embd ""your/path/to/nli-roberta-large" \
-d "your/path/to/truthful_qa" \
-S 0 \
-L 0 \
-b 4 \
--gpu0 0 \
--gpu1 1 \
-r 0.6 \
--truth "your/path/to/Truth-Llama-31-8B" \
--info "your/path/to/Info-Llama-31-8B"
