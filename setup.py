from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tf2crf',
    version='0.1.6',
    description='a crf layer for tensorflow 2 keras',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT License',
    url='https://github.com/xuxingya/tf2crf',
    author='xingya.xu',
    author_email='xingya.xu@gmail.com',
    install_requires=['tensorflow>=2.1.0', 'tensorflow_addons>=0.8.2'],
    packages=find_packages()
)