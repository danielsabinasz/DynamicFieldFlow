from setuptools import setup

setup(name='DynamicFieldFlow',
      version='0.1',
      description='A library for simulating Dynamic Field architectures with TensorFlow',
      url='https://github.com/danielsabinasz/DynamicFieldFlow',
      author='Daniel Sabinasz',
      author_email='daniel@sabinasz.net',
      license='CC-BY-ND 3.0',
      packages=['dff'],
      install_requires=[
          'tensorflow',
          'matplotlib',
          'numpy',
          'tensorflow_probability'
      ],
      zip_safe=False)
