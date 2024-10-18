## Impact of calibration data

### Setting 1: w4a16g128 llama2-7b seq_len=512

#### Calibrate with wikitext2

|           | wikitext2 | c4    | ptb    |
| --------- | --------- | ----- | ------ |
| GPTQ      | **5.575** | 7.470 | 63.575 |
| AWQ       | **5.595** | 7.444 | 35.167 |
| OmniQuant | **5.586** | 7.455 | 34.192 |

#### Calibrate with c4

|           | wikitext2 | c4        | ptb     |
| --------- | --------- | --------- | ------- |
| GPTQ      | 5.615     | **7.443** | 122.070 |
| AWQ       | 5.596     | **7.436** | 33.148  |
| OmniQuant | 5.620     | 7.457     | 34.001  |

#### Calibrate with pileval

|           | wikitext2 | c4    | ptb    |
| --------- | --------- | ----- | ------ |
| GPTQ      | 5.610     | 7.477 | 136.84 |
| AWQ       | 5.613     | 7.438 | 33.18  |
| OmniQuant | 5.618     | 7.458 | 34.526 |

### Setting 2: w3a16g128 llama2-7b seq_len=512

#### Calibrate with wikitext2

|           | wikitext2 | c4    | ptb     |
| --------- | --------- | ----- | ------- |
| GPTQ      | **6.133** | 8.696 | 234.977 |
| AWQ       | **6.138** | 8.272 | 38.86   |
| OmniQuant | **6.096** | 8.325 | 40.667  |

#### Calibrate with c4

|           | wikitext2 | c4        | ptb     |
| --------- | --------- | --------- | ------- |
| GPTQ      | 6.324     | **8.385** | 358.013 |
| AWQ       | 6.181     | **8.249** | 39.27   |
| OmniQuant | 6.259     | **8.317** | 41.835  |

#### Calibrate with pileval

|           | wikitext2 | c4    | ptb     |
| --------- | --------- | ----- | ------- |
| GPTQ      | 6.330     | 8.534 | 263.279 |
| AWQ       | 6.217     | 8.284 | 37.117  |
| OmniQuant | 6.214     | 8.320 | 42.335  |
