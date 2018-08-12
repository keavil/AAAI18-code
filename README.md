# Learning Structured Representation for Text Classification via Reinforcement Learning
Tianyang Zhang*, Minlie Huang, Li Zhao

Representation learning is a fundamental problem in natural language processing. This paper studies how to learn a structured representation for text classification. Unlike most existing representation models that either use no structure or rely on pre-specified structures, we propose a reinforcement learning (RL) method to learn sentence representation by discovering optimized structures automatically. We demonstrate two attempts to build structured representation: Information Distilled LSTM (ID-LSTM) and Hierarchically Structured LSTM (HS-LSTM). ID-LSTM selects only important, task-relevant words, and HS-LSTM discovers phrase structures in a sentence. Structure discovery in the two representation models is formulated as a sequential decision problem: current decision of structure discovery affects following decisions, which can be addressed by policy gradient RL. Results show that our method can learn task-friendly representations by identifying important words or task-relevant structures without explicit structure annotations, and thus yields competitive performance.

@inproceedings{zhang2018learning,

  title={Learning Structured Representation for Text Classification via Reinforcement Learning},
  
  author={Zhang, Tianyang and Huang, Minlie and Zhao, Li},
  
  booktitle={AAAI},
  
  year={2018}
  
}

AGnews dataset used in the experiment:
https://drive.google.com/open?id=1becf7pzfuLL7qgWqv4q-TyDYjSzodWfR
