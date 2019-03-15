"""TODO: this doesnt work, fix"""

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    return continuous_chunk

my_sent = "Westminster Abbey, formally titled the Collegiate Church of Saint Peter at Westminster, is a large, " \
           "mainly Gothic abbey church in the City of Westminster, London, England, just to the west of the Palace of " \
           "Westminster. It is one of the United Kingdom's most notable religious buildings and the traditional place " \
           "of coronation and burial site for English and, later, British monarchs. "

print get_continuous_chunks(my_sent)
