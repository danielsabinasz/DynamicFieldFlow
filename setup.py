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
          'tensorflow',
          'matplotlib',
          'numpy',
          'tensorflow-probability'
      ],
      zip_safe=False)
