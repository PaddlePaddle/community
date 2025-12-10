## 文心大模型4.5适配认证标准

<table border="2" style="width: 1500px;">
	<tr >
		<td width="10%" rowspan="2">分级</td>
		<td colspan="1">模型 </td>
		<td colspan="2">SFT</td>
                <td colspan="2">SFT-LoRA</td>
		<td colspan="2">强化学习</td>
		<td colspan="4">推理-集中式</td>
	</tr>
	<tr >
		<td> 针对模型发证 </td>
                <td> 性能 </td>
		<td> 上下文长度 </td>
                <td> 性能 </td>
		<td> 上下文长度 </td>
		<td>  </td>
		<td>  </td>	
		<td> 针对模型发证 </td>
                <td> 数据类型支持 </td>
                <td> 性能 </td>
		<td> 上下文长度 </td>
	</tr>
	<tr >
		<td>I级</td>
		<td rowspan="3">ERNIE-4.5-424B-A47B<br><br>ERNIE-4.5-28B-A3B<br><br>ERNIE-4.5-300B-A47B<br><br>ERNIE-4.5-21B-A3B<br><br>ERNIE-4.5-0.3B</td>
		<td >无要求</td>
		<td >8&32K</td>
		<td>无要求</td>
		<td >8&32K</td>
		<td> </td>
		<td> </td>
		<td rowspan="3">ERNIE-4.5-424B-A47B<br><br>ERNIE-4.5-28B-A3B<br><br>ERNIE-4.5-300B-A47B<br><br>ERNIE-4.5-21B-A3B<br><br>ERNIE-4.5-0.3B</td>
		<td>WINT8</td>
		<td>无性能要求</td>
		<td>8&32K</td>

</tr>
</tr>
	<td>II级</td>
	<td >性能达到benchmark的50%</td>
	<td>8&32&128K</td>
	<td >性能达到benchmark的50%</td>
	<td>8&32&128K</td>
	<td></td>
	<td></td>
	<td>WINT4</td>
	<td>单机TPS达到benchmark的50%</td>
	<td>8&32&128K</td>
</tr>	
</tr>
	<td>III级</td>
	<td>性能达到benchmark的80%</td>
	<td>8&32&128K</td>
	<td>性能达到benchmark的80%</td>
	<td>8&32&128K</td>
	<td></td>
	<td></td>
	<td>W4A8</td>
	<td>单机TPS达到benchmark的80% </td>
	<td>8&32&128K</td>
</tr>	
</tr>
	<td>验收说明</td>
	<td colspan="5">

- benchmark的硬件环境是A100/A800<br>
- 在指定有监督数据集上，在同样的训练超参数配置下（随机种子、学习率策略、全局批大小等），400步后（到500步）loss曲线逐位diff（绝对误差）基本维持在+-1e-2以内，outlier不多于10个，且diff无发散迹象。<br>
- 模型性能：基于前500step计算平均token/s</td>
  
  <td colspan="2"></td><br>
  <td colspan="4">
模型效果：<br>
- 纯文：模型在指定数据集（GSM8K+其他3个不同领域数据集）评测，得分与GPU结果持平(相差1分以内)；。<br>
- 多模：模型在指定数据集（MMLU+其他3个不同领域数据集）评测，得分与GPU结果持平(相差1分以内)；<br>
- 效果基准：联系<br>


模型性能测试说明：<br>

- 测试数据集介绍
- 文本：5000条，基于shareGPT进行抽取。<br>
- 图片：2350条，基于OCRBench_v2数据集进行抽取。<br>
- 300B&424B：在解码速度20、首token时延<1s（多模<4s）的约束条件下，测试性能。<br>
- 21B&28B：在解码速度40、首token时延<300ms（多模<4s）的约束条件下，测试性能。<br>
- 0.3B：在解码速度60、首token时延<300ms的约束条件下，测试性能。</td>
  </tr>
</table>


## Notes
-  [1]AI芯片厂商与文心大模型4.5的适配证书基于以上分级标准进行测试颁发证书
-  [2]一体机厂商与文心大模型4.5的适配证书基于以上分级标准的基础级别（I级）进行测试颁发证书