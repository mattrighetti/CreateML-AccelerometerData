# Create ML with Accelerometer Data

Simple playground code that generates a CoreML model that recognized when a user is **sitting**, **walking**, **standing** to be used in iOS and MacOS applications 

## Accelerometer Data

Accelerometer data can be found in the training.csv with different labels

- Label 1: Data collected when a person was **sitting**
- Label 2: Data collected when a person was **walking**
- Label 3: Data collected when a person was **standing**

## Known issues

- Due to the small dataset gathered (as little as 5 minutes of data recording) the model's predictions are unstable sometimes