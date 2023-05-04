from setuptools import setup, find_packages

classifiers = [
    "Intend Audience :: Education",
    "Licence :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3"
]

setup(
    name= "codes_pkges" ,
    version="0.0.1" ,
    description= "Contains usuals functions for data analyze",
    #long_description= open("README.md").read(),
    url= "https://github.com/Altagrac/DATA_PACKAGES",
    author= "Esperance Asngar",
    author_email="hoperancy@gmail.com",
    license= "MIT",
    classifiers= classifiers,
    keywords="" ,
    packages=find_packages(),
    install_requires=[""]
)
