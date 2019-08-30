import re

# The below is adapted from Robert MacIntyre's sed script for tokenising text. 
# https://github.com/andre-martins/TurboParser/blob/master/scripts/tokenizer.sed
# Ported to python by Matthew Farrugia-Roberts for COMP90049 Web Search and Text
# Analysis project, semester 1, 2019
# Adjusted for dealing with twitter data for COMP90051 Statistical Machine
# Learning project, semester 2, 2019
# ---------------------------------------------------------------------------- #

def sed_tokenize(line):
    for tokenizer, replacement in TOKENIZERS:
        line = tokenizer.sub(replacement, line)
    return line.split()

TOKENIZERS = [ (re.compile(r'^"'), r'`` ')
             # s=^"=`` =g
             
             # s=\([ ([{<]\)"=\1 `` =g
             , (re.compile(r'([ ([{<])"'), r'\g<1> `` ')
             
             # s=\.\.\.= ... =g
             , (re.compile(r'\.\.\.'), r' ... ')
             
             # s=[,;:@#$%&]= & =g
             # s=[?!]= & =g
             , (re.compile(r'[,;:@#$%&!?]'), r' \g<0> ')

             # # Assume sentence tokenization has been done first, so split
             # # FINAL periods only.
             # s=\([^.]\)\([.]\)\([])}>"']*\)[     ]*$=\1 \2\3 =g
             # (PLUS MY OWN ADDITION: ------v detect period before '-LRB-' etc.
             , (re.compile(r'''([^.])([.])([-\]\)}>"']*)\s*$''')
               , r'\g<1> \g<2>\g<3> ')
             # # however, we may as well split ALL question marks and
             # # exclamation points, since they shouldn't have the abbrev.
             # # -marker ambiguity problem (SEE THE RULE TWO RULES ABOVE)

             # # parentheses, brackets, etc.
             # s=[][(){}<>]= & =g
             , (re.compile(r'[\]\[\(\){}<>]'), r' \g<0> ')
             # # Some taggers, such as Adwait Ratnaparkhi's MXPOST, use the 
             # # parsed-file version of these symbols.
             # s/(/-LRB-/g
             # s/)/-RRB-/g
             # s/\[/-LSB-/g
             # s/\]/-RSB-/g
             # s/{/-LCB-/g
             # s/}/-RCB-/g
             , (re.compile(r'\('), '-LRB-')
             , (re.compile(r'\)'), '-RRB-')
             , (re.compile(r'\['), '-LSB-')
             , (re.compile(r'\]'), '-RSB-')
             , (re.compile(r'{'),  '-LCB-')
             , (re.compile(r'}'),  '-RCB-')
             # I OBSERVED THIS IN WIKITITLES:
             , (re.compile(r':'),  '-COLON-')

             # s=--= -- =g
             , (re.compile(r'--'), ' -- ')
             # # First off, add a space to the beginning and end of each line,
             # # to reduce necessary number of regexps.
             # s=$= =
             , (re.compile(r'$'), ' ')
             # s=^= =
             , (re.compile(r'^'), ' ')

             # s="= '' =g
             , (re.compile(r'"'), " '' ")
             
             # # possessive or close-single-quote
             # s=\([^']\)' =\1 ' =g
             , (re.compile(r"([^'])' "), r"\g<1> ' ")
             # # as in it's, I'm, we'd
             # s='\([sSmMdD]\) = '\1 =g
             , (re.compile(r"'([sSmMdD]) "), r" '\g<1> ")
             # s='ll = 'll =g
             , (re.compile(r"'ll "), " 'll ")
             # s='re = 're =g
             , (re.compile(r"'re "), " 're ")
             # s='ve = 've =g
             , (re.compile(r"'ve "), " 've ")
             # s=n't = n't =g
             , (re.compile(r"n't "), " n't ")
             # s='LL = 'LL =g
             , (re.compile(r"'LL "), " 'LL ")
             # s='RE = 'RE =g
             , (re.compile(r"'RE "), " 'RE ")
             # s='VE = 'VE =g
             , (re.compile(r"'VE "), " 'VE ")
             # s=N'T = N'T =g
             , (re.compile(r"N'T "), " N'T ")

             # s= \([Cc]\)annot = \1an not =g
             , (re.compile(r" ([Cc])annot "), r" \g<1>an not ")
             # s= \([Dd]\)'ye = \1' ye =g
             , (re.compile(r" ([Dd])'ye "), r" \g<1>' ye ")
             # s= \([Gg]\)imme = \1im me =g
             , (re.compile(r" ([Gg])imme "), r" \g<1>im me ")
             # s= \([Gg]\)onna = \1on na =g
             , (re.compile(r" ([Gg])onna "), r" \g<1>on na ")
             # s= \([Gg]\)otta = \1ot ta =g
             , (re.compile(r" ([Gg])otta "), r" \g<1>ot ta ")
             # s= \([Ll]\)emme = \1em me =g
             , (re.compile(r" ([Ll])emme "), r" \g<1>em me ")
             # s= \([Mm]\)ore'n = \1ore 'n =g
             , (re.compile(r" ([Mm])ore'n "), r" \g<1>ore 'n ")
             # s= '\([Tt]\)is = '\1 is =g
             , (re.compile(r" '([Tt])is "), r" '\g<1> is ")
             # s= '\([Tt]\)was = '\1 was =g
             , (re.compile(r" '([Tt])was "), r" '\g<1> was ")
             # s= \([Ww]\)anna = \1an na =g
             , (re.compile(r" ([Ww])anna "), r" \g<1>an na ")
             # # s= \([Ww]\)haddya = \1ha dd ya =g
             , (re.compile(r" ([Ww])haddya "), r" \g<1>ha dd ya ")
             # # s= \([Ww]\)hatcha = \1ha t cha =g
             , (re.compile(r" ([Ww])hatcha "), r" \g<1>ha t cha ")
             
             # # clean out extra spaces
             # s=  *= =g
             , (re.compile(r'\s\s*'), ' ')
             # s=^ *==g
             , (re.compile(r'^\s*'), '')
             ]
