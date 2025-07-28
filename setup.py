from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as ReadMe:
    long_description = ReadMe.read()

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'Operating System :: Microsoft :: Windows :: Windows 11',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11'
]

setup(
    name='GeoPatch',
    version='1.3.0',
    description='GeoPatch generates patches from remote sensing data and shapefiles for semantic segmentation and object detection with YOLO-format annotations',
    long_description=long_description,
    url='https://github.com/Hejarshahabi/GeoPatch',
    author='Hejar Shahabi',
    author_email='hejarshahabi@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='Machine Learning, Remote Sensing, Deep Learning, Object Detection, YOLO, Shapefile',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'rasterio',
        'geopandas',
        'shapely',
        'tqdm',
        'scikit-image'
    ]
)