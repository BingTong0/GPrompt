
## Quick Start
We have provided scripts with hyper-parameter settings to get the experimental results

In the pre-train phase, you can obtain the experimental results by running the parameters you want:
```shell
python pre_train.py --task Edgepred_Gprompt --dataset_name 'PubMed' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
```
or run `pre_train.sh`
```shell
cd scripts
./ pre_train.sh
```
In downstream_task, you can obtain the experimental results by running the parameters you want:

```shell
python downstream_task.py --pre_train_path 'None' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'None' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
```
or run `GraphTask.sh` for Graph task in **MUTAG** dataset, or run run `NodeTask.sh` for Node task in **Cora** dataset.




### Pre-train your GNN model

We have designed four pre_trained class (Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE), which is in ProG.pretrain module, you can pre_train the model by running ``pre_train.py`` and setting the parameters you want.

```python
import prompt_graph as ProG
from ProG.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE
from ProG.utils import seed_everything
from ProG.utils import mkdir, get_args


args = get_args()
seed_everything(args.seed)
mkdir('./pre_trained_gnn/')

pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)

pt.pretrain()



```
### Do the Downstreamtask
In ``downstreamtask.py``, we designed two tasks (Node Classification, Graph Classification). Here are some examples. 
```python
import prompt_graph as ProG
from ProG.tasker import NodeTask, LinkTask, GraphTask

if args.task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth', 
                    dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPrompt', epochs = 150, shot_num = 5)
    tasker.run()


if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
                    dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPrompt', epochs = 150, shot_num = 5)
    tasker.run()

```



  
**Kindly note that the comparison takes the same pre-trained pth.The absolute value of performance won't mean much because the final results may vary depending on different
  pre-training states.It would be more interesting to see the relative performance with other training paradigms.**

## Dataset

Our experiments are conducted on a diverse set of datasets, covering both node classification and graph classification tasks:

### Node Classification Datasets

| Dataset       | Description                                                                                                                       | Task                                  |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| **Cora, CiteSeer, and PubMed** | Citation networks where nodes represent documents, and edges represent citation links. Each node has a feature vector from the document text. | Classify nodes into academic topics.  |
| **Flickr**    | Social network dataset where nodes represent users, and edges represent follower relationships. Node features are based on user activity and metadata. | Classify users into interest groups.  |
| **ogbn-arxiv**| Citation network of computer science papers from arXiv, part of the Open Graph Benchmark. Nodes represent papers with feature vectors based on content. | Predict the subject area of each paper.|

### Graph Classification Datasets

| Dataset       | Description                                                                                                                       | Task                                  |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| **PROTEINS**  | Graphs representing proteins, with nodes as secondary structure elements and edges as spatial or sequential proximities.           | Classify proteins into categories.    |
| **COX2**      | Molecular graphs where nodes are atoms and edges are chemical bonds.                                                              | Predict the biological activity of molecules. |
| **ENZYMES**   | Graphs of enzyme structures.                                                                                                      | Predict the enzyme commission number. |
| **BZR**       | Molecular graphs.                                                                                                                 | Classify molecules based on biochemical properties. |
| **MUTAG**     | Graphs of mutagenic compounds, with nodes as atoms and edges as bonds.                                                            | Classify compounds by mutagenicity.   |
| **DD**        | Graphs representing protein structures, with nodes as amino acids and edges as interactions.                                      | Categorize proteins into structural families. |
| **COLLAB**    | Scientific collaboration graphs, with nodes as researchers and edges as co-authorships.                                           | Classify ego-networks into research fields. |

These datasets provide a comprehensive evaluation of our methods across various types of graphs and classification tasks, ensuring robustness and generalizability of the results.


