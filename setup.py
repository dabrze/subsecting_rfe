from distutils.core import setup

NAME = "brfe"
DESCRIPTION = "Bisecting Recursive Feature Elimination"
KEYWORDS = "Feature Selection Python"
AUTHOR = "Dariusz Brzezinski"
AUTHOR_EMAIL = "dariusz.brzezinski@cs.put.poznan.pl"
URL = "https://github.com/dabrze/bisecting_rfe"
VERSION = "1.0.0"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages =['brfe'],
    install_requires=["scikit-learn", "numpy", "scipy", "pandas", "seaborn",
                      "matplotlib"]
)
