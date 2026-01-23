# LangChain 最新版本 V1.x 中 Agent 的Human-in-the-Loop (HITL)

## 1、案例介绍

本期视频为大家分享的是如何在 LangChain 最新版本 V1.x 中实现Agent的Human-in-the-Loop (HITL)，也就是人机协作、人在环中、人工介入循环                                                             
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载                  

本期用例的核心功能包含：    

- 在invoke中使用HITL
- 在stream中使用HITL

### Human-in-the-Loop
      
HITL中间件可以在模型执行敏感操作(如写入文件或执行SQL)前暂停执行并等待人工审核           
该系统通过检查每个工具调用是否符合预设策略来决定是否需要干预,当需要时会发出中断信号暂停执行,利用LangGraph的持久化层保存图状态           

**人工审核者可以对中断做出三种响应：**     

- 批准(approve): 按原样执行操作,不做任何更改
- 编辑(edit): 修改参数后再执行工具调用
- 拒绝(reject): 拒绝该操作并提供反馈说明

**配置方式：**          

- 使用HITL需要在创建 Agent 时添加 HumanInTheLoopMiddleware 中间件,并配置 interrupt_on 映射来指定哪些工具需要人工审核以及允许的决策类型               
- 该功能必须配置检查点保存器(checkpointer)来持久化图状态                      


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
python agent_invoke_hitl.py
python agent_stream_hitl.py
```

{'city': '南京'}          
