from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4graph

args = get_args()
seed_everything(args.seed)


args.task = 'GraphTask'
args.prompt_type = 'Gprompt'

args.dataset_name = 'MUTAG'
args.pre_train_model_path = './Experiment/pre_trained_model/MUTAG/Edgepred_GPPT.GCN.128hidden_dim.pth'
args.shot_num = 5


input_dim, output_dim, dataset = load4graph(args.dataset_name)


if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, device=args.device, dataset = dataset, input_dim = input_dim, output_dim = output_dim)
    tasker.run()