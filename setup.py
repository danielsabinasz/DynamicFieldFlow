from setuptools import setup, find_packages
import distutils
is_mac = distutil.util.get_platform().startswith("mac")

if not is_mac:

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
              'numpy==1.22.3',
              'tensorflow-probability==0.15.0',
              'protobuf==3.20.0'
          ],
          zip_safe=False)

else:

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
              'tensorflow-macos==2.8.0',
              'matplotlib==3.5.1',
              'numpy==1.22.3',
              'tensorflow-probability==0.15.0',
              'protobuf==3.20.0'
          ],
          zip_safe=False)
