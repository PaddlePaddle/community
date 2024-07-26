## 飞桨芯片适配认证标准

<table border="2" >
	<tr >
		<td rowspan="2">硬件类型</td>
		<td rowspan="2">适配项目</td>
		<td colspan="4">适配要求</td>
	</tr>
	<tr >
		<td> I 级 </td>
		<td> II 级 </td>
		<td> III 级</td>
		<td> 成熟商用级 </td>
	</tr>
	<tr >
		<td width="10%" rowspan="6">AI训练芯片</td>
		<td>模型领域覆盖数量[1]</td>
		<td>2</td>
		<td>3</td>
		<td>4</td>
		<td>4</td>
	</tr>
		<td>模型数量[2] [9]</td>
		<td>2</td>
		<td>15</td>
		<td>30</td>
		<td>30</td>
	</tr>
	</tr>
		<td>算子种类</td>
		<td>60</td>
		<td>250</td>
		<td>350</td>
		<td>350</td>
	</tr>	
	</tr>
		<td>分布式训练</td>
		<td>单机单卡</td>
		<td>单机多卡</td>
		<td>多机多卡</td>
		<td>多机多卡</td>
	</tr>	
	</tr>
		<td>大模型</td>
		<td>无要求</td>
		<td>推理 I 级</td>
		<td>精调 I 级</td>
		<td>预训练 III 级</td>
	</tr>	
	</tr>
		<td>CI搭建</td>
		<td>无要求</td>
		<td>无要求</td>
		<td>覆盖编译+单测</td>
		<td>覆盖编译+单测</td>
	</tr>	
	<tr >
		<td width="10%" rowspan="3">AI推理芯片（数据中心）</td>
		<td>模型领域覆盖数量</td>
		<td>2</td>
		<td>3</td>
		<td>4</td>
		<td>[7]</td>
	</tr>
	</tr>
		<td>模型数量[10] </td>
		<td>2[5] /10[6] </td>
		<td>15[5] /50[6] </td>
		<td>50[5] /100[6] </td>
		<td>[7]</td>
	</tr>
	</tr>
		<td>算子种类</td>
		<td>30[5] /35[6]</td>
		<td>75</td>
		<td>175[5] /120[6] </td>
		<td>[7]</td>
	</tr>	
	<tr >
		<td width="10%" rowspan="6">AI推理芯片（移动/边缘计算）</td>
		<td>模型领域覆盖数量</td>
		<td>1</td>
		<td>2</td>
		<td>3</td>
		<td>[7]</td>
	</tr>
	</tr>
		<td>模型数量[10]</td>
		<td>3 </td>
		<td>20 （如支持量化模型，数量可降至10）</td>
		<td>50（如支持量化模型，数量可降至30） </td>
		<td>[7]</td>
	</tr>
	</tr>
		<td>算子种类</td>
		<td>20</td>
		<td>40</td>
		<td>75</td>
		<td>[7]</td>
	</tr>	
</table>

## Notes
- [1] 模型领域包括：视觉（分类、检测、分割）/OCR/NLP/时间序列 
- [2] 飞桨开源模型库包括大量经典模型和飞桨特色模型，每个模型有其所属领域，地址：https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta/docs/tutorials/models/support_model_list.md
- [3]基于全量数据集，端到端训推精度对齐
- [4] 基础训推功能验证
- [5] 以 Paddle Inference 适配
- [6] 以 Paddle Lite/ONNX/TVM 适配
- [7] 针对该类芯片，暂无此级别适配标准
- [8] 飞桨硬件适配全量算子列表：https://github.com/onecatcn/my-demo-code/blob/develop/PaddlePaddle/ops/gpu_ops_2023-03-20.csv
- [9] 训练精度要求：FP32 训练精度下误差小于正负 0.3%，AMP 混合精度训练下误差小于正负 3%，满足其中一个要求即可。
- [10] 推理精度要求：和 GPU/CPU 精度一致(移动边缘类芯片的量化模型预计有特殊损失，硬件厂商提供精度损失说明，由飞桨研发同学判断其合理性)
