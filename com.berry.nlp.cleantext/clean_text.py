# -*- coding: utf-8 -*-

from cleantext import clean

text = """A bunch of \\u2018new\\u2019 references, including [Moana](https://en.wikipedia.org/wiki/Moana_%282016_film%29).
                »Yóù àré     rïght &lt;3!« """

clean(text)