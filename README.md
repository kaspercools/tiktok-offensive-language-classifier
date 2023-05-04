# Detecting offensive content on TikTok

by
Kasper Cools, in partial fulfillment of the requirements for the degree of Master of Science in Computer Science

> This repository and the associated Python code and Jupyter notebooks are hereby published as part of my Master's
> thesis "Tick Tock, The clock is ticking. On the fine-tuning of Machine Learning models for offensive content
> classification on TikTok"

Supervisors:

- [Gideon Maillette de Buy Wenniger](https://scholar.google.nl/citations?user=7X7QIrgAAAAJ&hl=en)
- [Clara Maathuis](https://scholar.google.com/citations?user=WqR3BVwAAAAJ&hl=en)

## Abstract

> WiP

## Presentations

- April 19, 2023: [Young Talents International Conference 2023](http://ytic.eu)

## Software implementation

> This repository contains all python code used to train, test and evaluate our BERT model as well as our baseline
> Machine Learning models.
> The code, as-is, uses the HuggingFace [bert-base-uncased](https://huggingface.co/bert-base-uncased) model and training
> is configured to use 150 tokens max to match the needs for our particular use-case.

For the purpose of this research we wanted to measure the possible impact of adding custom tokens for specific Gen-Z
slang and emojis used on TikTok, therefore the training method receives 2 bool parameters to indcate if you want to add
emoji tokenization or slang. The emoji tokenization can be used as-is, but for slang you will need to either use
our [slang dataset](https://github.com/kaspercools/genz-dataset) or provide your own dictionary.

The F_SCORE_THRESHOLD const is used to limit the number of snapshots that are stored to disk during longer training
sessions. If the model F Score is lower than the given threshold, only the training results will be stored to disk. In
case the result exceeds the threshold, the results as well as a snapshot is saved to disk.

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/kaspercools/tiktok-offensive-language-classifier

## Datasets

Even though a vast amount of research has been performed in regards to the detection
of offensive language on social media, we did not find any datasets that suit our specific
use case. Most of the available datasets have focussed on data collected from platforms of
which the majority of users are not part of Gen-Z.
More specifically, the ideal dataset would contain data that is representative for the vast majority of
younger users on TikTok, especially those under the age of 27 (the Zoomer generation).

For the purpose of this research we collected a total of 3,138 TikTok video posts which subsequently were used to
collect a total of 120,423 comments over the course of 4 months (April 2022 to July 2022). Subsequently, these comments
were manually labelled resulting in a total of 78,181 which either contained English sentences, solely consisted of
emojis or more universally used expressions such as onomatopoeia. Of these 78,181 comments **2,034** were labelled
offensive.

Given the nature of the data, and taken into account
the [TOS of TikTok](https://www.tiktok.com/legal/page/eea/terms-of-service/en) we are not able to make our dataset
publicly available.
The dataset that can be found in the data folder is harvested
from [another github page/research](https://github.com/dhavalpotdar/detecting-offensive-language-in-tweets) and used to
evaluate our model's performance on unseen data.

## Docker setup

The easiest way to get started is by setting up your environment using docker.
To do so, you first need to build an image locally. Open a terminal window and navigate to the root of this repository
to execute the following command:
> docker build . -t ou_ml_tiktok

Once the docker image is built you can execute the python using the following command to start training:

``` dockerfile
docker run --rm -it --init \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  ou_ml_tiktok python3 src/main.py -i "data/comments_anonymous.csv"
````

### Running on a GPU

``` dockerfile
docker run --rm -it --init \
  --gpus=all
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  ou_ml_tiktok python3 src/main.py -i "data/comments_anonymous.csv"
````

## Environment setup
### Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the main folder to create a separate environment and install all required
dependencies in it:

    conda env create --name ENVIRONMENT_NAME
    conda activate ENVIRONMENT_NAME
    pip install -r requirements.txt

It is advised to perform any training of your own datasets on a GPU.
Note that ENVIRONMENT_NAME is an arbitrary name for your own reference so you can use any name you want.

## Running the code

If you wish, you can pass your own hyperparameters for fine-tuning the training process:

```    
    main.py 
        -i <inputfile> 
        -o <outputdir> 
        -l <learning rate> 
        -a <adam_epsilon>
        -v <validation ratio> 
        -e <epochs> 
        -b <batch size>
        -t <max token length> 
        -n <number of iterations>
        -m (includes emoji tokenization)
        -c <custom vocabulary file>
````

The only actual required parameter is the input csv dataset, the other values default to:

- batch_size = 32
- learning_rate = 5e-5
- adam_epsilon = 1e-08
- val_ratio = 0.2
- epochs = 2
- outputdir='models'
- dataFolder='data'
- iterations=100
- max_token_len = 150

Training sequence will look for cuda support, if cuda is not available, then cpu is used for training.

Another way of exploring and use the code is through jupyter notebook.
To do this, you must first start the notebook server by going into the
repository top level and running:
jupyter notebook

This will start the server and open your default web browser to the Jupyter
interface.

The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.

## Related submodules
The submodules that are added to this repo are some of the key scripts in collecting and processing the data in a (semi)automatic way. These were used to quickly and continuously scan, retrieve and collect data. Therefore, following submodules have been linked to this project:
- [genz-dataset](https://github.com/kaspercools/genz-dataset/tree/ffbb4f0594a3792e95de16f0243deef1b43c512c)
- [weaponized-word-collector](https://github.com/kaspercools/weaponized-word-collector)
- [bright-data-collector](https://github.com/kaspercools/bright-data-collector/tree/287979cf8cacb691fa39325aeb002d13c4ca9f15)
- [tiktok-selenium-crawler](https://github.com/kaspercools/tiktok-selenium-crawler/tree/e2f19e81ea44fdcb4054f04918e1c4447f4f6bdf)

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE` for the full license text.
