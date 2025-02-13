cd src
python -m task.mcqa \
-m "your/path/to/llm" \
-d "your/path/to/dataset" \
-S 0 \
-L 0 \
-b 4 \
--gpu 7 \
-r 0.6
