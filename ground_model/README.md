# Ground classifier
(see the paper for more details)

To run the scrip you will need the data.

The help text:
```bash
usage: learn.py [-h] [--vector {baseline,wPorts,ipsec}]
                [--sequence_length SEQUENCE_LENGTH]
                [--mode {binary,multiclass}] [--training_data IN_PATH]
                [--training_labels IN_PATH] [--test_data IN_PATH]
                [--test_labels IN_PATH] [--predict OUT_PATH] [--train]
                [--test] [--split_data] [--load_model IN_PATH]
                [--store_model OUT_PATH] [--verbose VERBOSE] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--plot OUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --vector {baseline,wPorts,ipsec}
                        which representation to use (default baseline)
  --sequence_length SEQUENCE_LENGTH
                        number of packets in te input sequence (default 20)
  --mode {binary,multiclass}
                        binary or multiclass classification (default binary)
  --training_data IN_PATH
                        source path of training data
  --training_labels IN_PATH
                        source path of training labels
  --test_data IN_PATH   source path of testing data
  --test_labels IN_PATH
                        source path of testing labels
  --predict OUT_PATH    make predictions from test data
  --train               train a new model
  --test                test the model
  --split_data          split the training data into training and testing
                        propotions
  --load_model IN_PATH  load a pretrained model
  --store_model OUT_PATH
                        save the model after training
  --verbose VERBOSE     verbose level (default 1)
  --epochs EPOCHS       number of epochs (default 128)
  --batch_size BATCH_SIZE
                        batch size during training (default 32)
  --plot OUT_PATH       generate plots of accuracy and loss

```

# Scenarios
## training a new binary model and test performance
Test data is available:
```bash
python learn.py --mode binary --training_data X_train.npy --training_labels y_train.npy --testing_data X_test.npy --testing_labels y_test.npy --train --test
```

Test data is not available:
```bash
python learn.py --mode binary --training_data X_train.npy --training_labels y_train.npy --train --test --split_data
```

## training a new multiclass model and generate predictions
Test data is available:
```bash
python learn.py --mode multiclass --training_data X_train.npy --training_labels y_train.npy --testing_data X_test.npy --testing_labels y_test.npy --train  --test --predict output_predictions.npy
```

Test data is not available:
```bash
python learn.py --mode multiclass --training_data X_train.npy --training_labels y_train.npy --train --test --split_data --predict output_predictions.npy
```

## load a pretrained model and test performance
```bash
python learn.py --testing_data X_test.npy --testing_labels y_test.npy --test --load_model input_model.h5
```


## training a new binary model with ipsec representation and 50 long-sequences then save the accuracy and loss plots
```bash
python learn.py --vector ipsec --sequence_length 50 --mode binary --training_data X_train.npy --training_labels y_train.npy --train --plot output_figures
```

# Contact
Fares Meghdouri
fares.meghdouri@tuwien.ac.at