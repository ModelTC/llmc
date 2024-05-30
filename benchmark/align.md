## Alignment with the Original Paper

### The conda environment is consistent with the requirements.txt file
### All other configurations are aligned with the original paper/code:

|             | calib_data | seq_len | num_data | seed |
|-------------|------------|---------|----------|------|
| GPTQ        | c4         | 2048    | 128      | 0    |
| AWQ         | pileval    | 512     | 128      | 42   |
| Omniquant   | wikitext2  | 2048    | 128      | 2    |
| Smoothquant | pileval    | 512     | 128      | 42   |
| Os_plus     | pileval    | 512     | 128      | 42   |

### Results

#### Weight-Only Asymmetric Quantization Results

|              | w4a16g128 | w3a16g128 | w2a16g64  |
|--------------|--------|--------|--------|
| GPTQ         | 5.623  | 6.318  | 14.968 |
| GPTQ-LLMC    | 5.623  | 6.318  | 14.968 |
| AWQ          |        |        |        |
| AWQ-LLMC     |        |        |        |
| Omniquant    | 5.590  | 6.092  | 9.525  |
| Omniquant-LLMC | 5.590  | 6.092  | 9.525  |

#### Weight-Activation Asymmetric Quantization Results

|               | w8a8 | w6a6 | w4a4  |
|---------------|--------|--------|--------|
| Omniquant     | 5.491  | 5.703  | 12.212 |
| Omniquant-LLMC | 5.490  | 5.703  | 12.239 |

#### Weight-Activation Symmetric Quantization Results

|               | w8a8  |
|---------------|-------|
| SmoothQuant   |       |
| SmoothQuant-LLMC |    |
| Os_plus       |       |
| Os_plus-LLMC  |       |
