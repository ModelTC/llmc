# 模型稀疏化

llmc目前正在逐渐支持稀疏化方法，目前已经实现了Magnitude和Wanda，将在未来支持更多的算法。

以下是Wanda的设置样例：


```
base:
    seed: &seed 42
model:
    type: Qwen2 # 设置模型名,可支持Llama,Qwen2,Llava,Gemma2等模型
    path: # 设置模型权重路径
    torch_dtype: auto
calib:
    name: pileval
    download: False
    path: # 设置校准数据集路径
    n_samples: 512
    bs: 1
    seq_len: 512
    preproc: pileval_smooth
    seed: *seed
eval:
    eval_pos: [pretrain, transformed] # 非结构化稀疏在稀疏过程中直接将对应位置权重置0，transformed之后直接就可以得到稀疏模型，无需再进行额外的部署阶段
    name: wikitext2
    download: False
    path: # 设置测试数据集路径
    bs: 1
    seq_len: 2048
sparse:
    method: Wanda
    weight:
        sparsity: 0.5 # 设置模型的稀疏率
    sparsity_out: False # 设置是否使用每一层稀疏后的输出作为下一层的输入
save:
    save_trans: True # 设置为True，可以保存下调整之后的权重
    save_path: ./save
```

以下展示了使用Wanda稀疏的一些结果：
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>www.lingdaima.com（零代码excel转HTML）</title>
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

在下面展示了与Wanda[原仓库](https://github.com/locuslab/wanda)对比的结果，在这一实验设置下，所使用的超参数、校准数据集以及数据预处理、评测方法均与Wanda仓库对齐。

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
