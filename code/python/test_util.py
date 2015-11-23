from util import count_links

BODY1 = """Nope. Don't buy it. 98% of all theory papers are done with massive input from all people. That's why the majority of the theory conferences list authors alphabetically. That's quite the contrast from real science work.

&gt;It can be brutally hard work that often involves long hours of (sometimes fruitless) attempts to prove something.

Hours? That's nothing. Try years.

&gt; And we do build things and design experiments,

Then it is not theoretical.

&gt;However, I can understand your confusion because theoreticians may spend a lot to time thinking, which to an untrained eye might look like ""doing nothing"".
"""

BODY2 = """join https://cstheory.stackexchange.com or http://cs.stackexchange.com and find out yourself! lots of good discussions in the chat rooms too."""

def test_count_links():
    assert count_links(BODY1) == 0
    assert count_links(BODY2) == 2

if __name__ == "__main__":
    test_count_links()
