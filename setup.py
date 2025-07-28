from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as ReadMe:
    long_description = ReadMe.read()

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'Operating System :: Microsoft :: Windows :: Windows 11',
    'Operating System :: POSIX :: Linux',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12'
]

setup(
    name='GeoPatch',
    version='1.3.2',
    description='GeoPatch generates patches from remote sensing data and shapefiles for semantic segmentation and object detection with YOLO-format annotations',
    long_description=long_description,
    url='https://github.com/Hejarshahabi/GeoPatch',
    author='Hejar Shahabi',
    author_email='hejarshahabi@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='Machine Learning, Remote Sensing, Deep Learning, Object Detection, YOLO, Shapefile, GeoTIFF',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=0.24.0',
        'rasterio>=1.2.0',
        'geopandas>=0.10.0',
        'shapely>=1.8.0',
        'tqdm>=4.60.0',
        'scikit-image>=0.18.0',
        'GDAL>=3.0.0'
    ],
    python_requires='>=3.8'
)