"""Sanity checks for the EPONYM_WHITELIST.

The whitelist is the source of truth shared by the InputSanitizer (do
not strip these names) and the AttributionGate (do not flag these
names). These tests catch:
  - Accidental removal of names already documented to be eponymous
  - Names that include trailing whitespace / typos
  - Frozenset immutability
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline import EPONYM_WHITELIST


REQUIRED_NAMES = [
    # Names that production-corpus sweeps confirmed as concept eponyms
    # appearing repeatedly in EPPP source material. Removing one will
    # cause the AttributionGate to start flagging that researcher.
    "Piaget", "Vygotsky", "Bowlby", "Erikson", "Kohlberg",
    "Pavlov", "Skinner", "Thorndike", "Tolman", "Premack",
    "Freud", "Jung", "Adler", "Rogers", "Maslow",
    "Beck", "Ellis", "Bandura",
    "Cannon", "Bard", "James", "Lange", "Yerkes", "Dodson",
    "Schachter", "Singer", "Selye", "Ekman",
    "Atkinson", "Shiffrin", "Tulving", "Baddeley",
    "Sapir", "Whorf", "Chomsky",
    "Asch", "Milgram", "Zimbardo", "Festinger",
    "Sherif", "Janis", "Latane", "Darley", "Berscheid",
    "Aronson", "Hovland", "Fiske", "Rosenthal", "Ajzen",
    "Sue", "Cross", "Berry", "Helms", "Ridley",
    "Kahneman", "Tversky", "Bem", "Bronfenbrenner", "Baumrind",
    "Holland", "Vroom", "Herzberg", "Fiedler", "Kirkpatrick",
    "Minuchin", "Lewinsohn", "Lazarus", "Caplan",
    "Broca", "Wernicke", "Korsakoff", "Alzheimer",
]


class TestEponymWhitelistContents(unittest.TestCase):
    def test_required_names_present(self):
        missing = [n for n in REQUIRED_NAMES if n not in EPONYM_WHITELIST]
        self.assertEqual(missing, [], f"Missing eponyms: {missing}")

    def test_no_whitespace_in_names(self):
        bad = [n for n in EPONYM_WHITELIST if n != n.strip()]
        self.assertEqual(bad, [], f"Names with surrounding whitespace: {bad}")

    def test_no_empty_strings(self):
        self.assertNotIn("", EPONYM_WHITELIST)

    def test_is_frozenset(self):
        self.assertIsInstance(EPONYM_WHITELIST, frozenset,
                              "Whitelist must be frozen to prevent runtime mutation")


if __name__ == "__main__":
    unittest.main()
