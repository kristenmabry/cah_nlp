#!/usr/bin/env python

import flask
import numpy as np
import cah

white_cards = []
def getWhiteCards():
    global white_cards
    global num_cards
    white_cards = np.random.choice(cah.cards['white'], 10 * num_cards)

black_card = []
num_cards = 1
def getBlackCard():
    global black_card
    global num_cards
    black_card = np.random.choice(cah.cards['black'], 1)[0]
    num_cards = max(black_card.count('___'), 1)

winner = -1
def refreshCards():
    global winner
    getBlackCard()
    getWhiteCards()
    winner = -1

# Create the application.
APP = flask.Flask(__name__)

@APP.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return flask.render_template('index.html',
        white_cards=white_cards,
        num_cards=num_cards,
        black_card=black_card,
        winner=winner
    )

@APP.route('/selectwinner', methods=['POST'])
def selectwinner():
    global winner
    winner = cah.getWinner(black_card, white_cards, num_cards)
    return flask.redirect('/')

@APP.route('/refresh', methods=['POST'])
def refresh():
    refreshCards()
    return flask.redirect('/')

if __name__ == '__main__':
    APP.debug=True
    refreshCards()
    APP.run(use_reloader=True, debug=True)