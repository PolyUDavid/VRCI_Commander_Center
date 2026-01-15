# 📋 Data Confidentiality Notice / 数据保密说明

**Version**: 1.0.0  
**Date**: January 15, 2026  
**Contact**: admin@gy4k.com

---

## English Version

### Training Data Confidentiality Statement

The actual training data used in the research paper *"Decentralizing Vehicle-Road-Cloud Integration: A Feasibility Study with AI-Enhanced Validation Platform and Sustainability Assessment"* cannot be publicly released due to the following reasons:

#### 1. Proprietary Laboratory Parameters

Our experimental simulation platform incorporates:
- **Fine-tuned system parameters** developed through extensive proprietary research
- **Specific scenario configurations** that reflect years of experimental optimization
- **Calibrated noise models** based on real-world deployment observations

These parameters represent significant intellectual property and competitive advantage for our research laboratory.

#### 2. Commercial Confidentiality Agreements

This research was conducted in collaboration with industry partners under binding Non-Disclosure Agreements (NDAs) that prevent disclosure of:
- **Operational deployment data** from real-world VRCI testbeds
- **Performance metrics** from proprietary vehicle communication systems
- **System design details** related to partner companies' commercial products

#### 3. Ethical and Legal Obligations

We are committed to:
- **Honoring contractual obligations** with research funders and industry collaborators
- **Protecting trade secrets** that could harm competitive positions
- **Maintaining trust** in academic-industry partnerships

---

### What IS Publicly Available

To maximize reproducibility while respecting confidentiality requirements, this repository provides:

#### ✅ Complete Model Architectures
- Exact layer definitions, dimensions, and activation functions
- All hyperparameters (learning rates, batch sizes, dropout rates)
- Training procedures (warmup, decay, early stopping)

#### ✅ Mathematical Formulations
- 67 equations in main paper and supplementary materials
- Complete derivations for all feasibility analyses
- Physical constraints and boundary conditions

#### ✅ Data Generation Methodology
- `generate_training_data.py`: Reconstructs datasets using public models
- Based on: M/M/1 queuing, path loss formulas, CMOS power laws
- Parameters from: 3GPP TS 22.186, ETSI TR 103 300-1, IPCC Guidelines

#### ✅ Evaluation Protocols
- Monte Carlo validation framework (500 iterations)
- Statistical analysis methods
- Confidence interval calculations

---

### Expected Performance with Generated Data

Models trained on the publicly available generated data should achieve:

| Metric | Paper Results | Expected Range with Generated Data |
|--------|--------------|-------------------------------------|
| **Latency Reduction** | 67.3% | 62-72% |
| **Energy Savings** | 42.7% | 38-48% |
| **Coverage Rate** | 95.7% | 92-97% |
| **Consensus Accuracy** | 96.9% | 94-98% |
| **Carbon Savings (10y)** | 2.2 kt | 2.0-2.5 kt |

**Key Points:**
- ✅ Qualitative trends will match (latency reduction with density, energy savings patterns)
- ✅ Order-of-magnitude performance will be preserved
- ⚠️ Exact numerical values may vary by ±5-10%
- ✅ Statistical significance of findings will be maintained

---

### Verification for Reviewers

For peer reviewers and editors requiring verification:

**Option 1: Aggregate Statistics**
We can provide aggregate statistics (mean, std, 95% CI) for key metrics without revealing raw data.

**Option 2: Third-Party Verification**
A trusted third party can verify results on the original dataset under NDA.

**Option 3: Supplementary Validation**
We can run reviewer-specified scenarios on our proprietary platform and report results.

**Contact**: admin@gy4k.com

---

## 中文版本 / Chinese Version

### 训练数据保密声明

论文《去中心化车路云一体化：基于AI增强验证平台和可持续性评估的可行性研究》中使用的实际训练数据由于以下原因无法公开发布：

#### 1. 实验室专有参数

我们的实验仿真平台包含：
- **经过微调的系统参数**：通过大量专有研究开发
- **特定场景配置**：反映多年实验优化的结果
- **校准的噪声模型**：基于真实部署观察

这些参数代表了我们研究实验室的重要知识产权和竞争优势。

#### 2. 商业保密协议

本研究与工业合作伙伴合作进行，受到具有约束力的保密协议（NDA）限制，禁止披露：
- **真实VRCI测试平台的运营部署数据**
- **专有车辆通信系统的性能指标**
- **与合作公司商业产品相关的系统设计细节**

#### 3. 伦理和法律义务

我们承诺：
- **履行合同义务**：对研究资助方和工业合作者
- **保护商业秘密**：避免损害竞争地位
- **维护信任**：在学术-产业合作伙伴关系中

---

### 公开提供的内容

为了在尊重保密要求的同时最大化可复现性，本仓库提供：

#### ✅ 完整的模型架构
- 精确的层定义、维度和激活函数
- 所有超参数（学习率、批量大小、dropout率）
- 训练流程（warmup、衰减、早停）

#### ✅ 数学公式
- 论文和补充材料中的67个方程
- 所有可行性分析的完整推导
- 物理约束和边界条件

#### ✅ 数据生成方法
- `generate_training_data.py`：使用公开模型重构数据集
- 基于：M/M/1排队论、路径损耗公式、CMOS功率定律
- 参数来自：3GPP TS 22.186、ETSI TR 103 300-1、IPCC指南

#### ✅ 评估协议
- Monte Carlo验证框架（500次迭代）
- 统计分析方法
- 置信区间计算

---

### 使用生成数据的预期性能

使用公开生成数据训练的模型应达到：

| 指标 | 论文结果 | 生成数据预期范围 |
|------|---------|-----------------|
| **延迟降低** | 67.3% | 62-72% |
| **能效提升** | 42.7% | 38-48% |
| **覆盖率** | 95.7% | 92-97% |
| **共识准确率** | 96.9% | 94-98% |
| **10年碳节约** | 2.2千吨 | 2.0-2.5千吨 |

**关键点：**
- ✅ 定性趋势将匹配（密度与延迟降低、能效模式）
- ✅ 数量级性能将保持
- ⚠️ 精确数值可能变化±5-10%
- ✅ 研究发现的统计显著性将保持

---

### 审稿人验证

对于需要验证的同行审稿人和编辑：

**选项1：汇总统计**
我们可以提供关键指标的汇总统计（均值、标准差、95%置信区间）而不透露原始数据。

**选项2：第三方验证**
可信第三方可以在NDA下对原始数据集验证结果。

**选项3：补充验证**
我们可以在专有平台上运行审稿人指定的场景并报告结果。

**联系方式**：admin@gy4k.com

---

## Legal Disclaimer / 法律免责声明

This data confidentiality approach complies with:
- **Bayh-Dole Act** (university technology transfer)
- **Trade Secrets Act** (proprietary information protection)
- **Standard research collaboration agreements**

本数据保密方法符合：
- **拜杜法案**（大学技术转让）
- **商业秘密法**（专有信息保护）
- **标准研究合作协议**

The authors affirm that all publishable results, conclusions, and validation methodologies are fully disclosed in the paper and this repository.

作者确认，所有可发表的结果、结论和验证方法均在论文和本仓库中完全公开。

---

## References / 参考文献

Similar approaches in published research:
1. Industry-academic collaborations in autonomous driving (Waymo, Tesla)
2. Telecommunications research with proprietary network data (Nokia, Ericsson)
3. Smart city projects with sensitive municipal data

类似的已发表研究方法：
1. 自动驾驶中的产学合作（Waymo、Tesla）
2. 涉及专有网络数据的通信研究（Nokia、Ericsson）
3. 涉及敏感市政数据的智慧城市项目

---

**This notice is referenced in:**
- `README.md` (Section: Data and Reproducibility)
- `docs/INSTALLATION.md` (Important Notice)
- `docs/REPRODUCIBILITY.md` (Critical Notice)
- `config/config_standard.yaml` (Header comments)
- `backend/training/generate_training_data.py` (Docstring)
- All training scripts (`train_*_model.py`)

**Document Version**: 1.0.0  
**Last Updated**: January 15, 2026  
**Maintained By**: NOK KO, Ma Zhiqin, Wei Zixian, Yu Changyuan  
**Contact**: admin@gy4k.com
