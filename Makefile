llama-3.2-1b-instruct:
	python main.py 												\
		--model_name_or_path  meta-llama/Llama-3.2-1B-Instruct	\
		--longbench

llama-3.2-3b-instruct:
	python main.py 												\
		--model_name_or_path  meta-llama/Llama-3.2-3B-Instruct	\
		--longbench

llama-3.1-8b-instruct:
	python main.py 												\
		--model_name_or_path  meta-llama/Llama-3.1-8B-Instruct	\
		--longbench

mistral-7b-instruct-v0.3:
	python main.py 												\
		--model_name_or_path mistralai/Mistral-7B-Instruct-v0.3	\
		--longbench

llama-2-13b-chat:
	python main.py 												\
		--model_name_or_path  meta-llama/Llama-2-13b-chat-hf	\
		--longbench