# LangChain 生态 V1.x 版本开发经验分享 

本开源项目会为大家分享关于LangChain V1.x、LangGraph V1.x、LangSmith及DeepAgents等开发经验                                
分享的全部视频涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，我会在每期视频中给大家提供免费的下载链接                   

**LangChain生态简介**          

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
- https://docs.langchain.com/oss/python/langchain/overview                               
- https://docs.langchain.com/oss/python/langgraph/overview                       
- https://docs.langchain.com/langsmith/home               

### 个人信息

1. YouTube频道(南哥AGI研习社)：https://www.youtube.com/channel/UChKJGiX5ddrIpJG-rBNVZ5g                                            
2. B站频道(南哥AGI研习社)：https://space.bilibili.com/509246474                      
3. GitHub地址：https://github.com/NanGePlus                    
4. Gitee地址：https://gitee.com/NanGePlus                                 
5. 大模型代理平台: https://nangeai.top/             
      
## 视频合集:

1. 2026 必学！LangChain 最新 V1.x 版本全家桶 LangChain + LangGraph + DeepAgents 开发经验免费开源分享
开源项目整体介绍     
https://youtu.be/W1js-VzhyiU     
https://www.bilibili.com/video/BV17c6mBbEHv/

2. 【EP01_快速入门用例】2026必学！LangChain最新V1.x版本全家桶LangChain+LangGraph+DeepAgents开发经验免费分享
https://youtu.be/0ixyKPE2kHQ   
https://www.bilibili.com/video/BV1EZ62BhEbR/   
对应文件夹:01_XXX

3. 【EP02_Prompt模版使用】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents开发经验免费分享          
https://youtu.be/xKnmdq-s2gc        
https://www.bilibili.com/video/BV1th69BAEdA/          
对应文件夹:02_XXX   

4. 【EP03_Agent 3 种调用方式】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents开发分享   
https://youtu.be/qYQY3WJ2KSU    
https://www.bilibili.com/video/BV1sjrBBnEy2/    
对应文件夹:03_XXX     

5. 【EP04_短期记忆持久化存储和记忆管理策略】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents分享        
https://youtu.be/rEhoJaNStzI         
https://www.bilibili.com/video/BV1i1kNBLEY3/                   
对应文件夹:04_XXX         

6. 【EP05_长期记忆持久化存储和读取写入记忆】2026必学！LangChain最新V1.x全家桶LangChain+LangGraph+DeepAgents分享              
https://youtu.be/rSe4GvIVNSA              
https://www.bilibili.com/video/BV1i1kNBLEY3/                   
对应文件夹:05_XXX    
