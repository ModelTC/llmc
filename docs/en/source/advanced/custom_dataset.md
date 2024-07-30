# Custom calibration datasets

Llmc currently supports the following types of calibration datasets.

1. pileval

2. wikitext2

3. c4

4. ptb

5. custom

where custom means that a user-defined calibration dataset is used. For some proprietary models in specific scenarios, it is more appropriate to use the data from that scenario for the calibration data when quantizing. Here's an example of a configuration.


```
calib:
    name: custom
    download: False
    load_from_txt: True
    path: # Custom dataset, ending with txt as suffix
    n_samples: 128
    bs: -1
    seq_len: 512
    preproc: random_truncate_txt
    seed: *seed
```

Users can write a piece of data text to a txt file, each line represents a piece of text data, using the above configuration, you can achieve the calibration of custom data sets.
