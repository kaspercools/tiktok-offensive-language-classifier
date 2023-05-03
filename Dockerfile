FROM anibali/pytorch:1.13.0-nocuda

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords
 