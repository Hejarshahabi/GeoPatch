from setuptools import setup, find_packages

with open("ReadMe.md", "r", encoding="utf-8") as ReadMe:
    long_description = ReadMe.read()

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='GeoPatch',
    version='1.1.1',
    description='GeoPatch is a package for generating patches from remote sensing data',
    long_description=long_description,
    url='https://github.com/Hejarshahabi',
    author='Hejar Shahabi',
    author_email='hejarshahabi@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='Machine Learning, Remote Sensing, Deep Learning',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    data_files=[('', ['GDAL-3.2.3-cp38-cp38-win_amd64.whl'])],
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'rasterio', 'tqdm'],
    
)
