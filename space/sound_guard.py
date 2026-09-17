"""Studio sound-description validation; no name lookup or lyric rewriting."""
import re
_DENY = re.compile('(?<![\\w-])kelly\\s+clarkson(?![\\w-])|(?<![\\w-])billie\\s+eilish(?![\\w-])|(?<![\\w-])slenderbodies(?![\\w-])|(?<![\\w-])chris\\s+cornell(?![\\w-])|(?<![\\w-])taylor\\s+swift(?![\\w-])|(?<![\\w-])soundgarden(?![\\w-])|(?<![\\w-])celine\\s+dion(?![\\w-])|(?<![\\w-])the\\s+beatles(?![\\w-])|(?<![\\w-])bruno\\s+mars(?![\\w-])|(?<![\\w-])audioslave(?![\\w-])|(?<![\\w-])mazzy\\s+star(?![\\w-])|(?<![\\w-])radiohead(?![\\w-])|(?<![\\w-])bon\\s+iver(?![\\w-])|(?<![\\w-])the\\s+xx(?![\\w-])|(?<![\\w-])jewel(?![\\w-])|(?<![\\w-])drake(?![\\w-])|(?<![\\w-])adele(?![\\w-])|(?<![\\w-])sade(?![\\w-])', re.IGNORECASE)
_SHORTHAND = re.compile("\\b[\\w'.]+(?:\\s+[\\w'.]+){0,4}\\s+[—–-]\\s+meaning\\b", re.IGNORECASE)
_APOSTROPHES = re.compile('[\\u2018\\u2019\\u02bc`]')

def validate_sound(text: str) -> None:
    folded = _APOSTROPHES.sub("'", text).casefold()
    if _SHORTHAND.search(text) or _DENY.search(folded):
        raise ValueError("Remove named music references; describe the sound directly")
