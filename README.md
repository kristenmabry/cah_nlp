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

Note: The model was too big to upload to github, so you'll have to train a new one yourself.

## Interface
- Go into the `interface` folder.
- Run `python app.py` to run the website.
- Go to `localhost:5000` to view and interact with the website.
- Press the `refresh` button to get a new round.
- Press the `show winner` button to highlight the model's chosen winner.

Note: Right now, the interface is set up to user the filtered, appropriate cards. This can be changed in `interface/cah.py` in line 9 by setting `isAppropriate` to `false`.


## Results
To view the raw results from my project, look at the `results.o` output file. This was the generated file from the Borah cluster when evaluating on the different datasets. Searching for `total:` will allow you to easily find the different accuracies. `Popular` refers to how many times the most popular card was chosen as the winner by the model.
