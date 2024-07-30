# Model Sparsification

The llmc is currently gradually supporting sparse methods, having already implemented Magnitude and Wanda, and will support more algorithms in the future.

Here is a sample of Wanda's settings:



```
base:
    seed: &seed 42
model:
    type: Qwen2 #  Set the model name, which can support Llama, Qwen2, Llava, Gemma2 and other models.
    path: # Set model weight path.
    torch_dtype: auto
calib:
    name: pileval
    download: False
    path: # Set calibration dataset path.
    n_samples: 512
    bs: 1
    seq_len: 512
    preproc: pileval_smooth
    seed: *seed
eval:
    eval_pos: [pretrain, transformed] # In the process of unstructured sparsification, the corresponding position weight is reset to 0 directly, and the sparse model can be obtained directly after transformed, without additional deployment stage
    name: wikitext2
    download: False
    path: # Set eval dataset path.
    bs: 1
    seq_len: 2048
sparse:
    method: Wanda
    weight:
        sparsity: 0.5 # Set model sparsity
    sparsity_out: False # Set whether use the output of the sparse layer as the input of the next layer.
save:
    save_trans: True # Set to True to save the adjusted weights.
    save_path: ./save
```

Here are some of the results of using Wanda:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title></title>
    <style>
table{border-top:1px solid #333;border-left:1px solid #333;border-spacing:0;background-color:#fff;width:100%}
table td{border-bottom:1px solid #333;border-right:1px solid #333;font-size:13px;padding:5px}
.et2{color:rgb(0, 0, 0);text-align:center;}
.et11{color:rgb(0, 0, 0);text-align:center ;}
.font0{color:rgb(0, 0, 0);}
</style>
</head>
<body>
    <table style="width:487.02pt"> 
 <colgroup>
  <col width="103" style="width:103.30pt;"> 
  <col width="48" style="width:48.00pt;" span="8"> 
 </colgroup>
 <tbody>
  <tr height="16"> 
   <td rowspan="3" class="et2">Model</td> 
   <td colspan="8" class="et2">PPL</td> 
  </tr> 
  <tr height="16"> 
   <td colspan="2" class="et11">dense</td> 
   <td colspan="2" class="et11">0.25</td> 
   <td colspan="2" class="et11">0.5</td> 
   <td colspan="2" class="et11">0.75</td> 
  </tr> 
  <tr height="16"> 
   <td class="et11">c4</td> 
   <td class="et11">wikitext2</td> 
   <td class="et11">c4</td> 
   <td class="et11">wikitext2</td> 
   <td class="et11">c4</td> 
   <td class="et11">wikitext2</td> 
   <td class="et11">c4</td> 
   <td class="et11">wikitext2</td> 
  </tr> 
  <tr height="16"> 
   <td class="et11">LLaMa2-7B</td> 
   <td class="et11">7.26</td> 
   <td class="et11">5.47</td> 
   <td class="et11">7.46</td> 
   <td class="et11">5.61</td> 
   <td class="et11">9.25</td> 
   <td class="et11">6.85</td> 
   <td class="et11">260.42</td> 
   <td class="et11">259.91</td> 
  </tr> 
  <tr height="16"> 
   <td class="et11">LLaMa2-70B</td> 
   <td class="et11">5.71</td> 
   <td class="et11">3.32</td> 
   <td class="et11">5.76</td> 
   <td class="et11">3.4</td> 
   <td class="et11">6.49</td> 
   <td class="et11">4.17</td> 
   <td class="et11">32.5</td> 
   <td class="et11">21.66</td> 
  </tr> 
  <tr height="16"> 
   <td class="et11">LLaMa3-8B</td> 
   <td class="et11">9.44</td> 
   <td class="et11">6.13</td> 
   <td class="et11">10.01</td> 
   <td class="et11">6.47</td> 
   <td class="et11">15.07</td> 
   <td class="et11">9.68</td> 
   <td class="et11">336.62</td> 
   <td class="et11">290.38</td> 
  </tr> 
  <tr height="16"> 
   <td class="et11">LLaMa3-70B</td> 
   <td class="et11">7.16</td> 
   <td class="et11">2.85</td> 
   <td class="et11">7.44</td> 
   <td class="et11">3.22</td> 
   <td class="et11">9.96</td> 
   <td class="et11">5.81</td> 
   <td class="et11">93.99</td> 
   <td class="et11">74.78</td> 
  </tr> 
 </tbody>
</table>
</body>
</html>

The results compared to origin [Wanda](https://github.com/locuslab/wanda) repository are shown below. In this experimental setup, the hyperparameters, calibration data sets, and data preprocessing and evaluation methods used are aligned with Wanda.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>www.lingdaima.com（零代码excel转HTML）</title>
    <style>
table{border-top:1px solid #333;border-left:1px solid #333;border-spacing:0;background-color:#fff;width:100%}
table td{border-bottom:1px solid #333;border-right:1px solid #333;font-size:13px;padding:5px}
.xl66{text-align:left;}
.xl65{text-align:center;}
</style>
</head>
<body>
    <table style="width:175pt"> <!--StartFragment--> 
 <colgroup>
  <col width="89" style="mso-width-source:userset;mso-width-alt:2848;width:67pt"> 
  <col width="72" span="2" style="width:54pt"> 
 </colgroup>
 <tbody>
  <tr height="20"> 
   <td class="xl66">Model</td> 
   <td class="xl65" data-width="200">Wanda</td> 
   <td class="xl65" data-width="200">LLMC</td> 
  </tr> 
  <tr height="39"> 
   <td class="xl66" data-width="200">LLaMa2-7b</td> 
   <td class="xl65" data-width="200">6.91</td> 
   <td class="xl65" data-width="200">6.91</td> 
  </tr> 
  <tr height="39"> 
   <td class="xl66" data-width="200">LLaMa2-70b</td> 
   <td class="xl65" data-width="200">4.22</td> 
   <td class="xl65" data-width="200">4.19</td> 
  </tr> 
  <tr height="39"> 
   <td class="xl66" data-width="200">LLaMa3-8b</td> 
   <td class="xl65" data-width="200">9.56</td> 
   <td class="xl65" data-width="200">9.58</td> 
  </tr> 
  <tr height="39"> 
   <td class="xl66" data-width="200">LLaMa3-70b</td> 
   <td class="xl65" data-width="200">OOM</td> 
   <td class="xl65" data-width="200">5.75</td> 
  </tr> <!--EndFragment--> 
 </tbody>
</table>
</body>
</html>
