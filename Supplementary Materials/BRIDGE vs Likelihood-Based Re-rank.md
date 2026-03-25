The following table reports the Damerau–Levenshtein distance (means and standard deviations) of BRIDGE and a re-ranking method that samples 50 suffixes 
and ranks them by likelihood (referred to as **Prob-Rank**). BRIDGE outperforms Prob-Rank, indicating that the performance gain cannot be attributed 
solely to the use of more random samples; instead, Bayes-risk-informed ranking plays a critical role.

|                   | Markov          |                 | LSTM            |                 | Transformer     |                 |
|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|                   | Prob-Rank       | BRIDGE          | Prob-Rank       | BRIDGE          | Prob-Rank       | BRIDGE          |
| NASA              |  0.5149±0.0006  |**0.4910±0.0011**|  0.5050±0.0026  |**0.4746±0.0051**|  0.5118±0.0102  |**0.4910±0.0110**|
| Sepsis            |  0.6312±0.0010  |**0.5200±0.0014**|  0.4493±0.0154  |**0.4264±0.0037**|  0.4600±0.0069  |**0.4307±0.0066**|
| BPIC12            |  0.6147±0.0003  |**0.5354±0.0005**|  0.5524±0.0035  |**0.5259±0.0039**|  0.5621±0.0029  |**0.5149±0.0054**|
| BPIC12-W          |  0.6363±0.0002  |**0.5322±0.0008**|  0.5891±0.0027  |**0.5676±0.0066**|  0.5908±0.0037  |**0.5449±0.0045**|
| BPIC13            |  0.2834±0.0015  |  0.2834±0.0016  |**0.2289±0.0029**|  0.2307±0.0040  |**0.2619±0.0027**|  0.2715±0.0047  |
| BPIC17            |  0.5833±0.0001  |**0.5379±0.0002**|  0.4501±0.0043  |**0.4409±0.0054**|  0.4572±0.0014  |**0.4286±0.0041**|
| BPIC19            |  0.3400±0.0001  |**0.3387±0.0001**|  0.2977±0.0029  |**0.2964±0.0025**|  0.2996±0.0012  |**0.2963±0.0020**|
| BPIC20            |  0.3166±0.0006  |**0.3064±0.0005**|**0.2042±0.0021**|  0.2101±0.0028  |**0.1996±0.0054**|  0.2020±0.0061  |
