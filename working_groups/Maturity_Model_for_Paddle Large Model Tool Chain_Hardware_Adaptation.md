##飞桨大模型工具链适配认证标准


<table border="2" >
	<tr >
		<td width="10%" rowspan="2">分级</td>
		<td colspan="2">调优 : SFT + LoRA</td>
		<td colspan="2">预训练 Pretrain</td>
		<td rowspan="2">DPO</td>
		<td colspan="3">推理 Inference</td>
	</tr>
	<tr >
		<td> 模型 </td>
		<td> 性能要求 </td>
		<td> 模型 </td>
		<td> 性能要求 </td>
		<td> 模型 </td>
		<td> 数据类型支持 </td>
		<td> 性能要求 </td>	
	</tr>
	<tr >
		<td>I级</td>
		<td >LLaMA1-13B</td>
		<td rowspan="3">无</td>
		<td rowspan="3">LLaMA1-13B</td>
		<td>tokens/TFLOPS （取前1000步均值）达到A100/800的20%</td>
		<td rowspan="4">待建设</td>
		<td> LLaMA1-13B</td>
		<td>FP16/ BF16</td>
		<td>首token 时延不超过1s的QPS/TFPLOPs达到A800的20%</td>
	</tr>
	</tr>
		<td>II级</td>
		<td >Qwen2-14B <br>SD(SFT only)</td>
		<td>tokens/TFLOPS （取前1000步均值）达到A100/800的40%</td>
		<td>Qwen2-14B<br>SD</td>
		<td> int8 （weight only）</td>
		<td>首token 时延不超过1s的QPS/TFPLOPs达到A800的40%</td>
	</tr>	
	</tr>
		<td>III级</td>
		<td>LLaMA3-70B <br>Qwen2-57B-A14B<br>(SFT only)</td>
		<td>tokens/TFLOPS （取前1000步均值）达到A100/800的60%</td>
		<td>LLaMA3-70B<br>GPT-3-175B（只看性能）<br>Qwen2-57B-A14B</td>
		<td> PTQ int8 （int8 * int8）<br>int4（weight only）</td>
		<td>首token 时延不超过1s的QPS/TFPLOPs达到A800的40%</td>
	</tr>	
	</tr>
		<td>验收要求</td>
		<td colspan="2"> 模型效果：<br>
在指定有监督数据集上按给定超参数上完成精调（SFT、LoRA两种精调场景）后，通过无随机性的贪心搜索解码生成方式，在给定 验证集上用ROUGE指标进行评测，与基准加速卡比较，效果指标与GPU结果持平( ± 1%以内)，人工评估结果与GPU结果持平。</td>
		<td colspan="2">训练精度：<br>
在百GB级别语料按照指定学习率、BatchSize，总步数，最大序列长度等超参后启动预训练任务<br>
•初期模型精度验证：给定初始模型下训练，去除训练随机性，在前1000步训练Loss中每20步取平均值，与GPU训练结果对比相对误差持平;<br>
•后期模型精度验证：收敛后模型在指定验证集上评估Loss，与GPU相比绝对误差需<1e-2；<br>
模型效果：<br>
•收敛后模型在指定数据集评测，准确率与GPU结果持平(± 1%以内)；<br>
训练性能：<br>
•每TFLOPS处理的tokens数量（取前1000步均值），达到百度提供benchmark作为基线（各级要求见上）；<br>
稳定性:<br>
•从启动预训练到完成训练任务中无出现宕机情况；<br>
•如完成任务时间过长，需保证至少连续14天多机训练不宕机；<br>
•训练期间如遇宕机按照给定Checkpoint热启后Loss无突刺可稳定下降；</td><br>
		<td colspan="3">模型效果：<br>
* 模型在指定数据集评测，准确率与GPU结果持平(± 1%以内)，人工评估结果与GPU结果持平；

推理性能：
•每TFLOPS处理的QPS（首token 时延不超过1s），达到百度提供benchmark作为基线（各级要求见上）。</td>
	</tr>
</table>
## Notes
-  [1]每一级是在较低一级的基础上增加模型要求，预训练认证需要满足同级的调优认证要求。
-  [2]LLM类别推荐适配开源模型列表：

| 模型 | 代码地址 | 
|:------|:-------:|
| GPT-3 | https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/config/gpt-3 | 
| LLaMA | https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/config/llama |
| Qwen | https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen |
-  文生图类别推荐适配开源模型：SD https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/README.md





