cd src
python -m task.qa \
-m "your/path/to/llm" \
--nli "your/path/to/deberta-v3-large-mnli" \
--embd "your/path/to/nli-roberta-large" \
--method "base" \
-d "your/path/to/dataset" \
-S 0 \
-L 0 \
-b 4 \
-r 0.4
