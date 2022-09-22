from setuptools import setup, find_packages

setup(name='DynamicFieldFlow',
      version='0.1',
      description='A library for simulating Dynamic Field architectures with TensorFlow',
      url='https://github.com/danielsabinasz/DynamicFieldFlow',
      author='Daniel Sabinasz',
      author_email='daniel@sabinasz.net',
      license='CC-BY-ND 3.0',
      packages=find_packages(include=['dff', 'dff.*']),
      install_requires=[
          'DynamicFieldPy',
          'tensorflow==2.8.0',
          'matplotlib==3.5.1',
          'numpy==1.21',
          'protobuf=3.20',
          'tensorflow-probability==0.15.0'
      ],
      zip_safe=False)
