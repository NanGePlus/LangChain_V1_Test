# LangChain 最新版本 V1.x 中集成 Langfuse 服务

## 1、案例介绍

本期视频为大家分享的是如何在 LangChain 最新版本 V1.x 中集成 Langfuse 服务                                                                                                                
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载                    

本期用例的核心功能包含：    

- Langfuse本地服务部署  
- LangChain 代码中集成 Langfuse
- Observability 观测功能测试
- Prompt Management 提示词管理功能测试
- Evaluation 评估功能测试

### Langfuse介绍

Langfuse 是一个开源的 LLM 工程平台，帮助团队协作式地对其 LLM 应用进行调试、分析和迭代。所有平台特性都原生整合在一起，用于加速开发工作流       

官网地址:https://langfuse.com/docs            

#### 1.1 Observability（可观测性）   

可观测性对于理解和调试 LLM 应用至关重要，与传统软件不同，LLM 应用包含复杂且非确定性的交互，这使得监控和调试变得更加困难             
Langfuse 提供全面的追踪能力，帮助你准确了解应用中正在发生的一切              

- 追踪信息包含所有 LLM 与非 LLM 调用，包括检索、向量嵌入、API 调用等
- 支持将多轮对话作为会话进行跟踪，并进行用户级别的追踪
- 可以将智能体（Agent）表示为图结构进行可视化
- 可通过原生的 Python/JS SDK、50+ 库/框架集成     
- 基于 OpenTelemetry 构建，以提升兼容性并减少对特定厂商的锁定风险              

OpenTelemetry（简称 OTel）是目前云原生领域最重要、最活跃的可观测性（Observability）标准和工具集合，几乎已经成为现代分布式系统监控的“事实标准”           
OpenTelemetry 是一个开源的、厂商中立的、可观测性框架，用于生成、采集、处理和导出以下三种核心遥测信号（Telemetry Signals）：                   

- Traces（链路追踪 / 分布式追踪）      
- Metrics（指标 / 度量） 
- Logs（日志）  

**核心概念：**           

**（1）Traces（追踪）**            

一个 trace 通常对应一次完整请求/操作        
比如用户向聊天机器人提一个问题到拿到回复的整个过程，包含整体输入输出以及用户、session、标签等元数据                     

**（2）Observations（观测）**            

trace 中的最小“步骤”，代表一次 LLM 调用、工具调用、RAG 检索等，支持嵌套形成层级结构，方便还原复杂 agent 调用链

**（3）Sessions（会话）**            

可选，用来把多个相关 trace 归为一次“会话”或工作流，比如一个聊天线程         
适合多轮对话或者复杂流程，官方建议在这类应用中使用 session                    

**属性与标注**

在有了 trace 与 observation 之后，可以通过不同“属性”给数据打标，方便筛选和分析       

**（1）Environments**         

区分 production / staging / development 等部署环境     

**（2）Tags**

灵活标签，用于按功能、接口、workflow 等维度分类 trace

**（3）User**

记录触发该 trace 的终端用户

**（4）Metadata**

任意键值对，用来附加自定义信息

**（5）Releases & Versions**

记录应用版本和组件变更，便于回溯问题与对比版本效果

#### 1.2 Prompt Management（Prompt管理）

Prompt Management 是对 Prompt 进行集中存储、版本管理和检索的系统化方案，用于 LLM 应用               
通过把 Prompt 放在 Langfuse 中，而不是代码里，产品和领域专家可以直接在平台web端中修改Prompt，工程师只需在应用里按 key 获取最新版本       
在 Langfuse 中，一个 prompt 对象 = 给 LLM 的指令（单条字符串或多轮消息数组）+ 可选的配置（如模型参数等）           

提供 Chat 与 Text 两种 Prompt：   

- Text prompt：单字符串，适合简单场景或只需要系统消息的场景      
- Chat prompt：由带角色的消息数组组成（system/user/assistant），适合完整对话结构、示例对话和携带历史记录的复杂应用       

**主要优势**        

**（1）解耦Prompt更新与代码发布**  

Prompt的迭代由非技术人员完成，不需要每次修改都走一遍开发、代码评审和上线流程，Prompt更新可以几乎“即时上线”

**（2）无额外延迟和可用性风险**

Langfuse SDK 会在客户端做缓存，读取Prompt几乎等同于从内存读取，不会显著增加请求延迟 

**进阶用法**       

- 将 prompt 与 traces 关联，按不同版本分析效果表现             
- 使用版本控制和标签来区分不同环境（例如开发、预发、生产）中的部署状态 
- 变量、缓存、A/B 测试等

#### 1.3 Evaluation（评估）

评估对于确保 LLM 应用的质量与可靠性至关重要   
Langfuse 提供灵活的评估工具，可根据你的具体需求进行调整，无论是在开发环境测试还是在生产环境监控表现      

- 可以使用多种评估方法入门：LLM-as-a-judge、用户反馈、人工标注或自定义方式
- 通过在生产 Trace 上运行评估，及早发现问题
- 创建和管理数据集，在开发阶段进行系统化测试，确保应用在不同场景下表现稳定可靠 
- 运行实验，以系统性地测试你的 LLM 应用

**评估闭环：离线 vs 在线**

**（1）离线评估**

在部署前，用固定数据集跑实验（Experiments），比如对一组测试用例运行新 Prompt 或模型，看分数和输出再迭代，然后才上线

**（2）在线评估**

对线上真实流量的 trace 自动打分，发现数据集里没有覆盖的边缘案例，再把这些案例补回数据集，形成持续改进闭环

**评估方法（Evaluation Methods）**

评估方法就是给 trace、observation、session 或 dataset run 打分的函数，支持多种方式

**（1）LLM-as-a-Judge**

用 LLM 按自定义标准打分，适合大规模主观指标（语气、准确性、帮助度等）

**（2）Scores via UI**

在 Langfuse UI 里手动给 trace 打分，用于快速抽查和单条排查

**（3）Annotation Queues**

结构化人工标注流程，用队列管理标注任务，适合做 ground truth、系统化标注和团队协作

**（4）Scores via API/SDK**

通过 API/SDK 程序化打分，适合自定义评估流水线、规则/确定性检查和自动化工作流

**（5）Score Analytics**

配套专门工具用来分析和校验这些分数


## 2、准备工作

### 2.1 集成开发环境搭建  

anaconda提供python虚拟环境,pycharm提供集成开发环境          

具体参考如下视频:                         
【大模型应用开发-入门系列】集成开发环境搭建-开发前准备工作                         
https://www.bilibili.com/video/BV1nvdpYCE33/                    
https://youtu.be/KyfGduq5d7w                        

### 2.2 大模型LLM服务接口调用方案

(1)gpt大模型等国外大模型使用方案                   
国内无法直接访问，可以使用Agent的方式，具体Agent方案自己选择                         
这里推荐大家使用:https://nangeai.top/register?aff=Vxlp          

(2)非gpt大模型方案 OneAPI方式或大模型厂商原生接口          

(3)本地开源大模型方案(Ollama方式)            

具体参考如下视频:                                                      
【大模型应用开发-入门系列】大模型LLM服务接口调用方案                   
https://www.bilibili.com/video/BV1BvduYKE75/              
https://youtu.be/mTrgVllUl7Y                           


## 3、项目初始化

关于本期视频的项目初始化请参考本系列的入门案例那期视频，视频链接地址如下:             

【EP01_快速入门用例】2026必学！LangChain最新V1.x版本全家桶LangChain+LangGraph+DeepAgents开发经验免费分享             
https://youtu.be/0ixyKPE2kHQ                    
https://www.bilibili.com/video/BV1EZ62BhEbR/               

### 3.1 下载源码

大家可以在本期视频置顶评论中获取免费资料链接进行下载              
 
### 3.2 构建项目 

使用pycharm构建一个项目，为项目配置虚拟python环境                                            
项目名称：LangChainV1xTest                                                                                     
虚拟环境名称保持与项目名称一致                                                      
 
### 3.3 将相关代码拷贝到项目工程中         

将下载的代码文件夹中的文件全部拷贝到新建的项目根目录下                             

### 3.4 安装项目依赖                

新建命令行终端，在终端中运行如下指令进行安装               
  
```bash
pip install langchain==1.2.1
pip install langchain-openai==1.1.6   
pip install concurrent-log-handler==0.9.28     
pip install langgraph-checkpoint-postgres==3.0.2 
pip install langchain-text-splitters==1.1.0 
pip install langchain-community==0.4.1
pip install langchain-chroma==1.1.0
pip install pypdf==6.6.0
pip install mcp==1.25.0  
pip install langchain-mcp-adapters==0.2.1
pip install langfuse==3.12.0
```

**注意:** 建议先使用这里列出的对应版本进行本项目脚本的测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                                 


## 4、功能测试   
                                
### 4.1 使用Docker方式运行Langfuse    

进入官网 https://github.com/langfuse/langfuse/tree/main 或通过命令                  
`git clone https://github.com/langfuse/langfuse.git` 下载安装源码             

打开命令行终端执行 `docker-compose up -d` 运行 langfuse 服务                         
运行成功后可在浏览器中访问 http://localhost:3000                        

### 4.2 运行脚本测试            

```bash
python create_index.py
python mcp_start.py
python agent_rag.py
```

Langfuse中配置human_prompt的config参数：             
{     
   "response_format": {       
     "type": "json_schema",      
     "json_schema": {     
       "name": "weather_schema",    
       "schema": {    
         "type": "object",    
         "properties": {     
           "punny_response": { "type": "string" },     
           "weather_conditions": { "type": "string" }     
         },     
         "required": ["punny_response", "weather_conditions"],      
         "additionalProperties": false     
       },     
       "strict": true      
     }      
   }         
 }         

