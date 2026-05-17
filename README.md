# 零基础上手LangChain V1.x 实战： 学最主流Agent开发框架

## 本系列视频定位

本系列内容将带领大家从零基础理解 LangChain V1.x 及其相关生态（包括 LangGraph、DeepAgents 等）在当前主流智能体（Agent）开发中的定位与核心能力。将依次完成基础环境准备、依赖安装、入门级 LLM 调用、Prompt 工程实战、Agent 类自动化流程构建、知识库（RAG）、向量检索、结构化/非结构化数据处理、多模态与插件集成等能力点。通过真实业务场景（如问答助手、内容摘要、API 组装等）的逐步拆解与落地案例，配合全流程源码讲解和操作演示，你将掌握 LangChain 最新 V1.x 体系下的核心开发范式，以及最佳工程实践和故障排查思路，最终能够独立搭建、升级和维护自己的智能体应用与工作流系统。

### 本系列仓库位置

以下仓库存储我在YouTube频道和B站频道关于零基础上手LangChain V1.x实战相关分享所有源文件，均开源免费  
此仓库对应的GitHub地址: [https://github.com/NanGePlus/LangChain_V1_Test](https://github.com/NanGePlus/LangChain_V1_Test)  
此仓库对应的Gitee地址: [https://gitee.com/NanGePlus/LangChain_V1_Test](https://gitee.com/NanGePlus/LangChain_V1_Test)                         

### 我的个人信息

- YouTube频道(@南哥AGI研习社)：[https://www.youtube.com/channel/UChKJGiX5ddrIpJG-rBNVZ5g](https://www.youtube.com/channel/UChKJGiX5ddrIpJG-rBNVZ5g)
- B站频道(@南哥AGI研习社)：[https://space.bilibili.com/509246474](https://space.bilibili.com/509246474)
- GitHub地址：[https://github.com/NanGePlus](https://github.com/NanGePlus)
- Gitee地址：[https://gitee.com/NanGePlus](https://gitee.com/NanGePlus)
- 大模型代理平台: [https://nangeai.top/](https://nangeai.top/)

### 其他开源分享推荐

- **n8n系列**：最新n8n自动化工作流平台  
B站视频链接：[https://www.bilibili.com/video/BV1Aq1NBYELp/](https://www.bilibili.com/video/BV1Aq1NBYELp/)  
YouTube视频链接：[https://www.youtube.com/playlist?list=PL8zBXedQ0uflhkZBwlQNAp7H57CJFgfgV](https://www.youtube.com/playlist?list=PL8zBXedQ0uflhkZBwlQNAp7H57CJFgfgV)  
GitHub地址：[https://github.com/NanGePlus/N8NWorkflowsTest](https://github.com/NanGePlus/N8NWorkflowsTest)  
Gitee地址：[https://gitee.com/NanGePlus/N8NWorkflowsTest](https://gitee.com/NanGePlus/N8NWorkflowsTest)                    
- **OpenClaw系列**：从零打造智能体驱动的商业自动化闭环  
B站视频链接：[https://www.bilibili.com/video/BV1svQGBBERQ/](https://www.bilibili.com/video/BV1svQGBBERQ/)  
YouTube视频链接：[https://www.youtube.com/playlist?list=PL8zBXedQ0ufmtUvaHsSxNqZMwgb3hxsJB](https://www.youtube.com/playlist?list=PL8zBXedQ0ufmtUvaHsSxNqZMwgb3hxsJB)  
GitHub地址：[https://github.com/NanGePlus/OpenClawTutorial](https://github.com/NanGePlus/OpenClawTutorial)    
Gitee地址：[https://gitee.com/NanGePlus/OpenClawTutorial](https://gitee.com/NanGePlus/OpenClawTutorial)                 
- **Cursor系列**：专为提升开发生产力而设计的一款AI提效工具   
B站视频链接：[https://www.bilibili.com/video/BV1HEABzvEdo/](https://www.bilibili.com/video/BV1HEABzvEdo/)     
YouTube视频链接：[https://www.youtube.com/playlist?list=PL8zBXedQ0ufkcPJWHVKFTFzS8yNCkfM5b](https://www.youtube.com/playlist?list=PL8zBXedQ0ufkcPJWHVKFTFzS8yNCkfM5b)     
- **更多开源项目**        
GitHub地址：[https://github.com/NanGePlus](https://github.com/NanGePlus)        
Gitee地址：[https://gitee.com/NanGePlus](https://gitee.com/NanGePlus)     

---

## 本系列视频链接地址速查

**【开源项目整体介绍】2026 必学！LangChain 最新 V1.x 版本全家桶 LangChain + LangGraph + DeepAgents 开发经验免费开源分享**

- YouTube频道对应视频: [https://youtu.be/W1js-VzhyiU](https://youtu.be/W1js-VzhyiU)     
- B站频道对应视频: [https://www.bilibili.com/video/BV17c6mBbEHv/](https://www.bilibili.com/video/BV17c6mBbEHv/)

**（1）【EP01_快速入门用例】2026必学！LangChain最新V1.x版本全家桶LangChain+LangGraph+DeepAgents开发经验免费分享**

- YouTube频道对应视频: [https://youtu.be/0ixyKPE2kHQ](https://youtu.be/0ixyKPE2kHQ)   
- B站频道对应视频: [https://www.bilibili.com/video/BV1EZ62BhEbR/](https://www.bilibili.com/video/BV1EZ62BhEbR/)   
- 对应文件夹:01_XXX

**（2）【EP02_Prompt模版使用】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents开发经验免费分享**          

- YouTube频道对应视频: [https://youtu.be/xKnmdq-s2gc](https://youtu.be/xKnmdq-s2gc)        
- B站频道对应视频: [https://www.bilibili.com/video/BV1th69BAEdA/](https://www.bilibili.com/video/BV1th69BAEdA/)          
- 对应文件夹:02_XXX

**（3）【EP03_Agent 3 种调用方式】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents开发分享**   

- YouTube频道对应视频: [https://youtu.be/qYQY3WJ2KSU](https://youtu.be/qYQY3WJ2KSU)    
- B站频道对应视频: [https://www.bilibili.com/video/BV1sjrBBnEy2/](https://www.bilibili.com/video/BV1sjrBBnEy2/)    
- 对应文件夹:03_XXX

**（4）【EP04_短期记忆持久化存储和记忆管理策略】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents分享**        

- YouTube频道对应视频: [https://youtu.be/rEhoJaNStzI](https://youtu.be/rEhoJaNStzI)         
- B站频道对应视频: [https://www.bilibili.com/video/BV1i1kNBLEY3/](https://www.bilibili.com/video/BV1i1kNBLEY3/)                   
- 对应文件夹:04_XXX

**（5）【EP05_长期记忆持久化存储和读取写入记忆】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents分享**              

- YouTube频道对应视频: [https://youtu.be/rSe4GvIVNSA](https://youtu.be/rSe4GvIVNSA)               
- B站频道对应视频: [https://www.bilibili.com/video/BV1DsrUBREsL/](https://www.bilibili.com/video/BV1DsrUBREsL/)                         
- 对应文件夹:05_XXX

**（6）【EP06_人工介入审查 HITL】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents分享**      

- YouTube频道对应视频: [https://youtu.be/hO2vNz_0mSs](https://youtu.be/hO2vNz_0mSs)       
- B站频道对应视频: [https://www.bilibili.com/video/BV191zzBaEjm/](https://www.bilibili.com/video/BV191zzBaEjm/)       
- 对应文件夹:06_XXX

**（7）【EP07_检索增强生成 RAG】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents分享**        

- YouTube频道对应视频: [https://youtu.be/xaxbaeMT0c0](https://youtu.be/xaxbaeMT0c0)        
- B站频道对应视频: [https://www.bilibili.com/video/BV18jzrBcEY3/](https://www.bilibili.com/video/BV18jzrBcEY3/)         
- 对应文件夹:07_XXX

**（8）【EP08_自定义MCP Server】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents分享**    

- YouTube频道对应视频: [https://youtu.be/Q9tJ5EY5-1Y](https://youtu.be/Q9tJ5EY5-1Y)      
- B站频道对应视频: [https://www.bilibili.com/video/BV1g3zDBxEUx/](https://www.bilibili.com/video/BV1g3zDBxEUx/)   
- 对应文件夹:08_XXX

**（9）【EP09_集成Langfuse服务】打造可观测可评估的Agent应用，从本地部署到观测评估的保姆级完整闭环攻略。可观测性、Prompt管理与评估一站式集成方案**                   

- YouTube频道对应视频: [https://youtu.be/WBaTb58E8Q4](https://youtu.be/WBaTb58E8Q4)            
- B站频道对应视频: [https://www.bilibili.com/video/BV1cv6iBqEFK/](https://www.bilibili.com/video/BV1cv6iBqEFK/)          
- 详细拆解测试：            
- YouTube频道对应视频: [https://youtu.be/Guvuf_xxdG0](https://youtu.be/Guvuf_xxdG0)              
- B站频道对应视频: [https://www.bilibili.com/video/BV13a6vBSEVT/](https://www.bilibili.com/video/BV13a6vBSEVT/)                
- 对应文件夹:09_XXX

**（10）【EP10_集成Milvus向量数据库】打造真正理解业务逻辑的企业级RAG应用，从服务本地部署、知识库构建、混合搜索、MCP到Agent调用的完整保姆级闭环攻略**                      

- YouTube频道对应视频: [https://youtu.be/1CCslws1mkA](https://youtu.be/1CCslws1mkA)      
- B站频道对应视频: [https://www.bilibili.com/video/BV1iLFazaEF4/](https://www.bilibili.com/video/BV1iLFazaEF4/)                      
- 对应文件夹:10_XXX

**（11）【EP11_API接口服务】基于FastAPI框架打造带人工审核+状态持久化的ReAct Agent API后端接口服务，完整开源保姆级闭环攻略，详细拆解测试**       

- YouTube频道对应视频: [https://youtu.be/kU5G8SdiwSA](https://youtu.be/kU5G8SdiwSA)   
- B站频道对应视频: [https://www.bilibili.com/video/BV1rSFozLEjF/](https://www.bilibili.com/video/BV1rSFozLEjF/)    
- 对应文件夹:11_XXX

**（12）【EP12_Skills】LangChain V1.x 集成Skills完整闭环攻略及详细功能拆解测试。支持按需加载Skill、说明即能力、工具化接入**               

- YouTube频道对应视频: [https://youtu.be/tjhOPa5msMU](https://youtu.be/tjhOPa5msMU)                   
- B站频道对应视频: [https://www.bilibili.com/video/BV1XqPdzuEQ3/](https://www.bilibili.com/video/BV1XqPdzuEQ3/)                 
- 对应文件夹:12_XXX

**（13）【EP13_打字机效果】后端直推前端：ReAct Agent API后端流式接口实现打字机效果完整闭环攻略和详细拆解测试。异步流式生成和SSE数据推流**     

- YouTube频道对应视频: [https://youtu.be/2Xf1LPR6mgY](https://youtu.be/2Xf1LPR6mgY)     
- B站频道对应视频: [https://www.bilibili.com/video/BV1dqwVzHESS/](https://www.bilibili.com/video/BV1dqwVzHESS/)                
- 对应文件夹:13_XXX

**（14）【EP14_Playwright】ReAct Agent API 后端接口实战：真实浏览器交互完整闭环与测试拆解。访问动态网页、提取文本余链接、点击可见元素**  

- YouTube频道对应视频: [https://youtu.be/Yw7Lx2nAdWQ](https://youtu.be/Yw7Lx2nAdWQ)    
- B站频道对应视频: [https://www.bilibili.com/video/BV14LQPBJEti/](https://www.bilibili.com/video/BV14LQPBJEti/)    
- 对应文件夹:14_XXX

**（15）【EP15_多Agents】ReAct Agent API 后端接口实战：多Agents智能体协作完整闭环保姆级攻略与测试拆解。人工审核、多轮中断、数据持久化**  

- YouTube频道对应视频: [https://youtu.be/QS6GeBin6Ew](https://youtu.be/QS6GeBin6Ew)    
- B站频道对应视频: [https://www.bilibili.com/video/BV1Jk5t6EEA7/](https://www.bilibili.com/video/BV1Jk5t6EEA7/)    
- 对应文件夹:15_XXX

**（16）【EP16_多Agents（LangGraph方案）】ReAct Agent API 后端接口实战：多Agents智能体协作完整闭环保姆级攻略与测试拆解。**  

- YouTube频道对应视频: [https://youtu.be/KhvaRtqU6b8](https://youtu.be/KhvaRtqU6b8)    
- B站频道对应视频: [https://www.bilibili.com/video/BV1vBLj6LEmz/](https://www.bilibili.com/video/BV1vBLj6LEmz/)    
- 对应文件夹:16_XXX

---

## LangChain生态简介

官方生态包括开源库(LangChain Python/JS)、底层编排框架 LangGraph、以及用于评测与观测的 LangSmith，形成从开发到上线的完整链路                   

现V1.x版本 LangChain 中的“Agent”是构建在 LangGraph 之上的，LangChain 提供上层易用 API，LangGraph 提供底层有状态、有分支/循环的图式执行引擎                     

LangChain 适合“快速起一个Agent/应用”的场景，当需要复杂控制流(循环、回滚、人工介入、长时对话状态)时，可直接下沉到 LangGraph 定义显式的状态机/有向图工作流                  

LangSmith 提供 trace、评测和对比试验工具，帮助团队在 prompt/参数/策略迭代中做数据驱动的优化与回归验证                 

**2025年10月20号发布v1.0版本,有两个主要变化：**                      

(1)对 langchain 中所有Chain和Agent进行了完全重构  
现在所有Chain和Agent都被统一为一个高级抽象：构建在 LangGraph 之上的Agent抽象  
这个Agent抽象最初是在 LangGraph 中创建的高级抽象，现在被迁移到了 LangChain 中                     

(2)标准化的消息内容格式  
模型 API 已经从只返回简单文本字符串的消息，演进为可以返回更复杂的输出类型，例如推理块、引用、服务器端工具调用等  
为此，LangChain 也相应演进了其消息格式，用于在不同模型提供商之间对这些输出进行标准化                

**官方文档**      

- [https://docs.langchain.com/oss/python/langchain/overview](https://docs.langchain.com/oss/python/langchain/overview)                               
- [https://docs.langchain.com/oss/python/langgraph/overview](https://docs.langchain.com/oss/python/langgraph/overview)                       
- [https://docs.langchain.com/langsmith/home](https://docs.langchain.com/langsmith/home)

---

## 适合的小伙伴

- 希望借助 LangChain 体系，将常见的重复劳动（如智能问答、内容摘要、API 组装、结构化抽取等）标准化为**可扩展、可维护的智能体工作流**的开发者、产品经理或成长型团队。  
- 需要整合多源异构数据或多模态输入（文本、网页、数据库、接口等），并能够自定义和维护**智能体驱动的自动化/半自动化服务**的小伙伴，尤其适合自托管和持续升级的实战场景。

