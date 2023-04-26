# Cards Against Humanity NLP Project

## Data Setup
- Request the data from [Cards Against Humanity Labs](https://lab.cardsagainsthumanity.com/).
- Save the csv file as `cah_2023.csv` in a new folder called `data`.
- Run the clean_data.ipynb notebook to clean up the data and separate into datasets. These will all be saved in the `data` folder.
- Go into the `scripts` folder and run `python split_data.py` to get train, validation, and test sets.

## Training
- Go into the `scripts` folder.
- Run `python train.py` to train a model. It will save the model in a folder called `cah_model`.
- Run `python eval.py` to get the accuracies for the different datasets.

## Interface
- Go into the `interface` folder.
- Run `python app.py` to run the website.
- Go to `localhost:5000` to view and interact with the website.
