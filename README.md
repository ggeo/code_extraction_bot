It's a simple app that can be run **locally** in your PC with the use of ollama.



You upload an image which contains code (works with python, may work with other language also) and it extracts the code for you.



The idea came because many people upload a code image in social instead of a text piece of code. So, this is a way to extract the code!



The app has 2 options. You can select `mistral` and run locally or `OPENAI` and run the openai chat but you need the openai api key `(gpt-4o-mini)`. In order to do so, you have to

`export OPENAI_API_KEY="your_openai_api_key_here"`



If you are going to use it locally, you need to [Download Ollama on Linux](https://ollama.com/download) and pull the mistral model `ollama pull mistral`.



Then, you have to start the ollama service `ollama start`.



The app uses `streamlit`, so you type `streamlit run chat.py`.



You need to install `sudo apt install tesseract-ocr` in your system.



In order to create the environment `conda env create -f environment.yml`.



I am including a couple of images to use in `data` folder.



You can see a demo.




https://github.com/user-attachments/assets/5c23c606-2ef7-4db7-8f6f-6fa79e719e66

