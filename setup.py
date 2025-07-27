from setuptools import setup, find_packages

with open("ReadMe.md", "r", encoding="utf-8") as ReadMe:
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
    version='1.2',
    description='GeoPatch generates patches from remote sensing data for semantic segmentation and object detection with YOLO-format annotations',
    long_description=long_description,
    url='https://github.com/Hejarshahabi/GeoPatch',
    author='Hejar Shahabi',
    author_email='hejarshahabi@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='Machine Learning, Remote Sensing, Deep Learning, Object Detection, YOLO',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    data_files=[('', ['GDAL-3.6.2-cp38-cp38-win_amd64.whl'])],
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'rasterio', 'tqdm', 'scikit-image']
)