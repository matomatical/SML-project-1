"""
functions for pre-processing tweets (split words/punctutation,
normalise sparse features like URLs, and more)
"""

import re
#                      v html escape code '&#?\w+;'
#                               v punctuation symbols [.,!?:;"'~(){}[]<>*=-]
PUNCT_GROUPS    = r"(((&#?\w+;)|[\.\,\!\?\:\;\"\'\~\(\)\{\}\[\]\<\>\*\=\-])+)"

# first separate punctuation groups from the end of each word
PUNCT_AT_END    = re.compile(PUNCT_GROUPS + r"\s"), r" \g<1> "
# then take it away from the start too :)
PUNCT_AT_START  = re.compile(r"\s" + PUNCT_GROUPS), r" \g<1> "
# if there is only one piece of punct in the middle (so "that's"
# or "wishy-washy" but not "https://example.com") then split that too!
PUNCT_IN_MIDDLE = re.compile(r"\s(\w+)" + PUNCT_GROUPS + r"(\w+)(?=\s)"), r" \g<1> \g<2> \g<5> "
# maybe we introduced some double spaces with the above; remove them.
EXTRA_SPACES    = re.compile(r"\s\s*"), r" "

def tokenise(tweet_text):
    # pad with spaces to simplify expressions (spaces are word boundaries)
    text = f" {tweet_text} "
    # apply a sequence of pattern substitutions
    sequence = [PUNCT_AT_END, PUNCT_AT_START, PUNCT_IN_MIDDLE, EXTRA_SPACES]
    for pattern, replacement in sequence:
        text = pattern.sub(replacement, text)
    return text

