import setuptools

# Parse requirements
install_requires = [line.strip() for line in open("requirements.txt").readlines()]

# Setup package
setuptools.setup(
    name="thegreatknowledgeheist",
    version="0.0.1",
    author="Michał Zobniów, Maria Wyrzykowska, Adrian Urbański ",
    python_requires=">=3.9",
    url="https://github.com/Sahcim/TheGreatKnowledgeHeist",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
)
