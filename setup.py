from setuptools import setup

NAME = "srfe"
DESCRIPTION = "Subsecting Recursive Feature Elimination"
KEYWORDS = "Feature Selection Python"
AUTHOR = "Dariusz Brzezinski"
AUTHOR_EMAIL = "dariusz.brzezinski@cs.put.poznan.pl"
URL = "https://github.com/dabrze/subsecting_rfe"
VERSION = "1.0.1"

with open('requirements.txt','r') as req_file:
    requirements = req_file.read().strip().splitlines()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=['srfe'],
    install_requires=requirements,
    build_requires=['numpy'],
    setup_requires=['numpy'],
)
