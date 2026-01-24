# LangChain 最新版本 V1.x 中 Agent 使用 MCP

## 1、案例介绍

本期视频为大家分享的是如何在 LangChain 最新版本 V1.x 中实现 Agent 的 MCP Server 调用                                                                           
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载                    

本期用例的核心功能包含：    

- 自定义 MCP Server：将 RAG 工具封装为一个 MCP Server 对外提供使用 
- Agent使用 MCP Server 

### MCP介绍
      
Model Context Protocol（MCP）是一个开放标准，用来规范「大模型如何安全、统一地调用外部工具和数据源」                   

- MCP 由 Anthropic 在 2024 年提出，被多家大模型厂商采用，用来统一「模型 ↔ 外部系统」的交互方式
- 底层基于 JSON-RPC 2.0，定义了一套通用消息格式，让模型可以发现工具、调用函数、读取资源和使用预设 prompt

**MCP 的核心概念**      

(1)Tools（工具） 

- MCP Server 可以暴露一组「可执行函数」，例如查数据库、调第三方 API、发请求到内部系统等
- 在 LangChain 中，这些 MCP tools 会被自动映射成 LangChain 的 Tool 对象，Agent 可以像用普通 Tool 一样调用

(2)Resources（资源）

- 用来暴露数据，如文件内容、数据库记录、HTTP 响应等，客户端可以以统一方式读取文本或二进制内容
- LangChain 会把这些资源转成 Blob 对象，方便后续做检索、解析或加载进上下文 

(3)Prompts（预设提示词）

- MCP 还支持让服务器提供一系列预设 prompt 模板，客户端或模型可以直接复用这些高质量提示词 

**MCP 与 LangChain 的集成**

- LangChain 通过 langchain-mcp-adapters 等库接入 MCP，提供 MCP 客户端（如 MultiServerMCPClient），可以同时连多个 MCP Server
- MCP tools 会被转成 LangChain 的 tools 列表，统一交给 Agent；MCP resources 会变成 Blob，用于读取上下文数据
- 传输层支持本地 stdio、SSE（Server-Sent Events）、HTTP Streamable，既适合本地开发，又能部署到云端服务

**相关链接** 

- MCP官方简介:https://www.anthropic.com/news/model-context-protocol             
- MCP文档手册:https://modelcontextprotocol.io/introduction           
- MCP官方服务器列表:https://github.com/modelcontextprotocol/servers           
- PythonSDK的github地址:https://github.com/modelcontextprotocol/python-sdk            

**开源项目**

- MCP开源项目:https://github.com/NanGePlus/MCPServerTest            

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
```

**注意:** 建议先使用这里列出的对应版本进行本项目脚本的测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                                 

## 4、功能测试   
                                
### 4.1 使用Docker方式运行PostgreSQL数据库     

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，`cd 04_ShortTermMemory/postgresql` 文件夹下                     
- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                            

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                       

使用数据库客户端软件远程登陆进行可视化操作，这里推荐使用免费的DBeaver客户端软件                     

- DBeaver 客户端软件下载链接: https://dbeaver.io/download/           
            
### 4.2 运行脚本测试            

```bash
python create_index.py
python mcp_start.py
python agent_rag.py
```

