"""
submission/stopwords.py — the shared stoplist.

Lives in its own module so that both the pure-Python tokenizer in
indexer.py and the compiled one in _fasttok.pyx can import it without an
import cycle (the compiled module is imported *by* indexer.py).

A compact, standard English stoplist. Deliberately does NOT strip
negations or "no"/"not": on a question-style topic set ("what causes death
from Covid-19") the function words carry no discriminative signal, but
polarity words occasionally do.
"""

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because
been before being below between both but by can cannot could did do does
doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself me
more most my myself of off on once only or other ought our ours ourselves
out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up
very was we were what when where which while who whom why with would you
your yours yourself yourselves
""".split())
