python3.8 preprocess_context.py --context_path "${1}" --data_path "${2}" --split test
python3.8 test_context.py --device cuda
python3.8 preprocess_QA.py --split test
python3.8 test_QA.py --device cuda --pred_file "${3}"