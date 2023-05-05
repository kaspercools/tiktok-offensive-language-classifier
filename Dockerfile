FROM anibali/pytorch:1.13.0-cuda11.8
WORKDIR /app

COPY ./requirements.txt ./
# install required python modules
RUN pip install --no-cache-dir -r requirements.txt
# download nltk stopwords
RUN python -m nltk.downloader stopwords
# dowload bert-base-uncased and save to cache, the dirty way
# if you read this and have a better way to do this, give me a ping! https://github.com/kaspercools
RUN python -c "__import__('transformers').AutoModel.from_pretrained('bert-base-uncased')"