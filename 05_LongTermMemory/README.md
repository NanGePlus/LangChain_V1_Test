# LangChain 最新版本 V1.x 中 Agent 的长期记忆

## 1、案例介绍

本期视频为大家分享的是如何在 LangChain 最新版本 V1.x 中实现Agent的长期记忆(跨对话线程持久化存储)功能，包括长期记忆的写入和查询                                                      
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载                  

本期用例的核心功能包含：    

- PostgresStore，基于数据库的长期记忆持久化存储       
- 写入长期记忆        
- 读取长期记忆  

### 长期记忆
      
短期记忆允许应用程序在单个对话线程(thread)内记住之前的交互。对话历史是最常见的短期记忆形式          
长期记忆，指的是跨对话、跨会话都能复用的、持久化存储的用户或应用信息，而不是只跟当前对话轮次绑定的“上下文窗口”         

**长期记忆是什么：**    

- 存什么：用户画像（名字、偏好、历史决策）、业务配置、历史任务结果等，需要“下次还能记得”的信息
- 作用范围：可以跨 thread / 会话使用，同一个用户在不同对话里都能被读出来，而不是只在单个聊天线程中可见

**存储位置与形式：**      

- LangChain / LangGraph 把长期记忆抽象为一个 Store（键值存储），可以是内存、Postgres、向量库等，实现统一接口 put / get / search
- 数据通常按 namespace + key 组织，比如 ["memories", user_id] 作为命名空间，再用 uuid 作为具体记忆条目的 key，value 是包含实际信息的字典   


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
python agent_PostgresStore.py
```


