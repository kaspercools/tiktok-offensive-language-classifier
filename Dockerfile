FROM anibali/pytorch:1.13.0-cuda11.8
WORKDIR /app

COPY ./requirements.txt ./
# install required python modules
RUN pip install --no-cache-dir -r requirements.txt
# download nltk stopwords
RUN python -m nltk.downloader stopwords