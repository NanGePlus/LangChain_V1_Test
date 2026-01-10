# LangChain 最新版本 V1.x 快速入门

## 1、介绍

本期视频为大家分享的是 LangChain 最新版本 V1.x 的快速入门用例                
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载         

本期用例的核心功能包含：

- 各厂商LLM大模型集成
- System Prompt 定义
- Tools 工具定义
- Models 数据模型定义
- Agent 定义和运行
- Agent 短期记忆(内存)
- Agent 工具调用
- Agent 结构化输出
- LangSmith 跟踪观测

### 什么是 Agents        

Agents 将 LLM 大语言模型与 Tools 工具结合，创建能够推理任务、决定使用哪些工具并迭代寻找解决方案的系统                
Agent遵循 ReAct ("推理 + 行动")模式，在满足停止条件前循环运行工具以实现目标             

官方介绍链接:https://docs.langchain.com/oss/python/langchain/agents              

#### 核心组件

1. 模型(Model) 

- 支持静态模型配置         
- 支持动态模型选择(通过中间件在运行时根据对话复杂度等因素切换模型)         

2. 工具(Tools) 

- 为Agent提供执行操作的能力
- 支持多个工具顺序调用、并行调用、动态工具选择及错误处理
- 可通过 @tool 装饰器自定义工具属性

3. 系统Prompt(System Prompt)
- 可使用字符串或 SystemMessage 定义Agent行为方式
- 支持动态系统Prompt

#### 高级功能

1. 结构化输出(2种策略)
- ToolStrategy: 通过人工工具调用生成结构化输出,适用于所有支持工具调用的模型
- ProviderStrategy: 使用模型提供商的原生结构化输出功能,更可靠但依赖提供商支持

2. 记忆(Memory)
- 通过消息状态自动维护对话历史
- 支持自定义状态模式以存储额外信息,作为Agent的短期记忆

3. 流式传输(Streaming)
- 支持通过 stream 方法实时返回中间步骤和消息

4. 中间件(Middleware)
- 在执行的不同阶段自定义Agent行为
- 可用于消息修剪、内容过滤、错误处理、动态模型选择等场景


## 2、准备工作

### 2.1 集成开发环境搭建  

anaconda提供python虚拟环境,pycharm提供集成开发环境          

具体参考如下视频:                         
【大模型应用开发-入门系列】集成开发环境搭建-开发前准备工作                         
https://www.bilibili.com/video/BV1nvdpYCE33/                    
https://youtu.be/KyfGduq5d7w                        

### 2.2 大模型LLM服务接口调用方案

(1)gpt大模型等国外大模型使用方案                   
国内无法直接访问，可以使用代理的方式，具体代理方案自己选择                         
这里推荐大家使用:https://nangeai.top/register?aff=Vxlp          

(2)非gpt大模型方案 OneAPI方式或大模型厂商原生接口          

(3)本地开源大模型方案(Ollama方式)            

具体参考如下视频:                                                      
【大模型应用开发-入门系列】大模型LLM服务接口调用方案                   
https://www.bilibili.com/video/BV1BvduYKE75/              
https://youtu.be/mTrgVllUl7Y                           

## 3、项目初始化

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
```

**注意:** 建议先使用这里列出的对应版本进行本项目脚本的测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                                   

## 4、功能测试   
                                            
- 运行脚本 `python agent.py`                                   
