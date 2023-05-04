FROM anibali/pytorch:1.13.0-nocuda
#FROM anibali/pytorch:1.13.0-cuda
WORKDIR /app

COPY ./requirements.txt ./
# install required python modules
RUN pip install --no-cache-dir -r requirements.txt
# download nltk stopwords
RUN python -m nltk.downloader stopwords