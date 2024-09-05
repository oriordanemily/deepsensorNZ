# Using DeepSensor to create high-resolution weather forecast fields over Aotearoa New Zealand.

The DeepWeather project (https://www.deepweather.org.nz/) aims to leverage machine learning methods to produce high-resolution weather forecasts for New Zealand as part of an operational weather forecast. The main DeepWeather repo, owned by [Bodeker Scientific](https://github.com/bodekerscientific), is currently private, but will be public when work is finalised. 

[DeepSensor](https://github.com/alan-turing-institute/deepsensor) is a Python package that implements convolutional neural processes (convNPs). ConvNPs are a deep-learning approach to Gaussian processes, making the most of their flexibility and uncertainty estimations but with a significant speed up in computational costs. 

In the DeepWeather project, we are using DeepSensor to create **initial high-resolution, observation-enhanced weather forecast fields**. A numerical weather prediction (NWP) model is typically initialised at some recent time in the past, to allow modellers to assimilate any observations that have been collected at that time into the forecast. We wish to downscale these initial forecast fields where observations are available, and use these observations to enhance the accuracy of our downscaling. DeepSensor allows us to feed in the lower-resolution gridded forecast field and any point-based observations, and creates a high-resolution gridded field with influence from these point observations. 

These fields can then be used both operationally and in training of the wider DeepWeather model, to produce high-resolution forecasts with the purpose of feeding this high-fidelity information back into the NWP model.


# Installing DeepSensorNZ

Once the repo has been cloned locally, create a conda environment using the environment.yml file. 

Navigate to the top of the repo and run pip install -e .
