#!/bin/env python

import os
import warnings

from transformers import pipeline, Conversation, Pipeline

warnings.filterwarnings("ignore")

GOAT = "🐐"

MODEL_NAME = "Rijgersberg/GEITje-7B-chat-v2"


def beeh(something: str):
    print(GOAT, something, sep=": ")


WELCOME = ("Welkom bij GEITje! "
           "Ik ben een vriendelijke AI Geit die gelukkig ook Nederlands kan. "
           "Verder houd ik erg van woordgrapjes en wortels!")


def loop(chatbot: Pipeline):
    conversation = Conversation()
    os.system("clear")
    beeh(WELCOME)
    conversation.add_message({"role": "assistant", "content": WELCOME})

    while prompt := input("> "):
        conversation.add_message({"role": "user", "content": prompt})
        conversation = chatbot(conversation)

        beeh(conversation[-1]['content'])


def main():
    chatbot = pipeline(
        task="conversational", model=MODEL_NAME, device_map="auto"
    )

    loop(chatbot)


if __name__ == "__main__":
    main()
